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
    create_surrogate_model,
)
from .analytical_models import get_analytical_model
from .profile_generator_fast import FastProfileGenerator
from .pvt_state import SolventExtendedPVTEngine, CO2_TONNE_PER_MSCF, MSCF_PER_TONNE
from .geomechanics_fault import GeomechanicsFaultModel

# Constants
_PHYS_CONSTANTS = PhysicalConstants()


def _safe_float(val: Any, default: float = 0.0) -> float:
    if val is None:
        return float(default)
    try:
        return float(val)
    except (ValueError, TypeError):
        return float(default)


def _safe_attr(obj: Any, attr: str, default: float = 0.0) -> float:
    if obj is None:
        return float(default)
    val = getattr(obj, attr, None)
    if val is None:
        return float(default)
    try:
        return float(val)
    except (ValueError, TypeError):
        return float(default)


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
        else:
            raise ValueError(f"Unknown model type: {model_type}. Deprecated non-analytical models removed; use 'analytical'.")

        # Baseline trapping parameters for profile generator
        default_trapping = {
            "structural": 0.15,
            "residual": 0.25,
            "solubility": 0.20,
            "mineral": 0.05,
            "total": 0.65,
        }
        # Initialize profile generator
        self.profile_generator = FastProfileGenerator(
            model_type=profile_model_type,
            trapping_params=default_trapping,
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

            # Map optimizer profile parameters to FastProfileGenerator expected names
            if "plateau_duration_fraction" in profile_params:
                profile_params["plateau_fraction"] = profile_params.pop("plateau_duration_fraction")
            if "hyperbolic_b_factor" in profile_params:
                profile_params["b_factor"] = profile_params.pop("hyperbolic_b_factor")

            profile_result = self.profile_generator.generate_profile(
                ooip=reservoir_data.ooip_stb,
                recovery_factor=recovery_factor,
                injection_rate=eor_params.injection_rate,
                project_lifetime=operational_params.project_lifetime_years,
                **profile_params
            )
            if "breakthrough_time_years" in params:
                profile_result["breakthrough_time_years"] = params["breakthrough_time_years"]

            # Calculate performance metrics
            evaluation_time = time.perf_counter() - start_time
            self.evaluation_count += 1
            self.total_evaluation_time += evaluation_time

            # Initialize Solvent-Extended Compositional PVT & Geomechanics Engines
            initial_pressure = _safe_attr(reservoir_data, 'initial_pressure', 3000.0)
            target_pressure = _safe_attr(eor_params, 'target_pressure_psi', 3200.0)
            time_vector = profile_result["time_vector"]
            dt = np.diff(time_vector, prepend=0)

            pvt_engine = SolventExtendedPVTEngine(
                reservoir_temperature_f=_safe_attr(reservoir_data, "temperature", _safe_attr(eor_params, "reservoir_temperature_f", 150.0)),
                initial_pressure_psi=initial_pressure,
                api_gravity=_safe_attr(reservoir_data, "api_gravity", 35.0),
                dead_oil_viscosity_cp=_safe_attr(eor_params, "default_oil_viscosity_cp", 2.0),
                c7_plus_fraction=_safe_attr(eor_params, "c7_plus_fraction", 0.35),
            )

            geomech_model = GeomechanicsFaultModel(
                depth_ft=_safe_attr(reservoir_data, "depth_ft", _safe_attr(eor_params, "reservoir_depth_ft", 5000.0)),
                initial_pressure_psi=initial_pressure,
                overburden_gradient_psi_per_ft=_safe_attr(eor_params, "overburden_gradient_psi_per_ft", 1.0),
                horizontal_stress_ratio_k0=_safe_attr(eor_params, "horizontal_stress_ratio_k0", 0.75),
                poissons_ratio=_safe_attr(eor_params, "poissons_ratio", 0.25),
                biot_coefficient=_safe_attr(eor_params, "biot_coefficient", 0.80),
                caprock_fracture_pressure_psi=_safe_attr(eor_params, "caprock_fracture_pressure_psi", 5500.0),
                caprock_tensile_strength_psi=_safe_attr(eor_params, "caprock_tensile_strength_psi", 200.0),
                caprock_cohesion_psi=_safe_attr(eor_params, "caprock_cohesion_psi", 400.0),
                caprock_friction_angle_deg=_safe_attr(eor_params, "caprock_friction_angle_deg", 30.0),
                caprock_safety_factor=_safe_attr(eor_params, "caprock_safety_factor", 0.90),
                fault_dip_deg=_safe_attr(eor_params, "default_fault_dip", _safe_attr(eor_params, "fault_dip_deg", 60.0)),
                fault_strike_deg=_safe_attr(eor_params, "default_fault_strike", _safe_attr(eor_params, "fault_strike_deg", 0.0)),
                fault_friction_coefficient=_safe_attr(eor_params, "default_fault_friction_coefficient", _safe_attr(eor_params, "fault_friction_coefficient", 0.60)),
                fault_cohesion_psi=_safe_attr(eor_params, "default_fault_cohesion", _safe_attr(eor_params, "fault_cohesion_psi", 0.0)),
            )

            if len(time_vector) > 0:
                ooip = _safe_attr(reservoir_data, 'ooip_stb', 1e6)
                swi = _safe_attr(reservoir_data, 'connate_water_saturation', _safe_attr(reservoir_data, 'initial_water_saturation', 0.25))
                
                # Dynamic reference FVF and pore volume
                bo_init = pvt_engine.calculate_oil_fvf_rb_per_stb(initial_pressure, 0.0)
                pore_volume_rb = (ooip * bo_init) / max(1.0 - swi, 0.05)
                
                # Injection gas FVF at initial conditions (RB/MSCF)
                b_co2_init = pvt_engine.calculate_co2_fvf_rb_per_mscf(initial_pressure)
                
                q_inj_rb = profile_result["injection_profile"] * b_co2_init
                q_prod_rb = (profile_result["oil_profile"] * bo_init +
                             profile_result["water_profile"] * 1.0 +
                             profile_result["gas_profile"] * 0.50)

                pressure_profile = np.zeros(len(time_vector))
                sat_oil_profile = np.zeros(len(time_vector))
                sat_gas_profile = np.zeros(len(time_vector))
                sat_water_profile = np.zeros(len(time_vector))
                bo_profile = np.zeros(len(time_vector))
                bg_profile = np.zeros(len(time_vector))
                viscosity_oil_profile = np.zeros(len(time_vector))
                fault_slip_profile = np.zeros(len(time_vector))
                caprock_margin_profile = np.zeros(len(time_vector))
                vrr_profile = np.zeros(len(time_vector))
                p_sandface_profile = np.zeros(len(time_vector))
                x_co2_profile = np.zeros(len(time_vector))
                y_co2_profile = np.zeros(len(time_vector))
                leakage_rate_profile = np.zeros(len(time_vector))

                # Calculate Peaceman well index base parameters
                nx_cells = _safe_float(params.get("nx"), 50.0)
                ny_cells = _safe_float(params.get("ny"), 50.0)
                length_val = _safe_attr(reservoir_data, "length_ft", _safe_float(params.get("length_ft"), 2000.0))
                area_val = _safe_attr(reservoir_data, "area_acres", _safe_float(params.get("area_acres"), 100.0))
                thickness_val = _safe_attr(reservoir_data, "thickness_ft", _safe_float(params.get("thickness_ft"), 50.0))
                avg_perm_val = _safe_attr(reservoir_data, "average_permeability", _safe_float(params.get("perm"), 100.0))

                dx_block = length_val / max(nx_cells, 1.0)
                width_val = (area_val * 43560.0) / max(length_val, 1.0)
                dy_block = width_val / max(ny_cells, 1.0)

                well_list = params.get("well_data_list") or getattr(reservoir_data, "well_data_list", None)
                if not well_list and hasattr(self, "well_data_list"):
                    well_list = self.well_data_list

                j_peaceman_prod_base = 0.0
                j_peaceman_inj_base = 0.0

                if well_list:
                    for well in well_list:
                        w_type = ""
                        if hasattr(well, "metadata") and isinstance(well.metadata, dict):
                            w_type = str(well.metadata.get("type", "")).lower()
                            if not w_type:
                                w_type = str(well.metadata.get("status", "")).lower()
                        if not w_type and hasattr(well, "name"):
                            w_type = "injector" if "inj" in well.name.lower() else "producer"

                        is_inj = "inj" in w_type
                        if hasattr(well, "calculate_peaceman_index"):
                            wi = well.calculate_peaceman_index(
                                k_mD=avg_perm_val,
                                h_ft=thickness_val,
                                dx_ft=dx_block,
                                dy_ft=dy_block,
                                mu_cp=1.0,
                            )
                        else:
                            r_o = 0.198 * np.sqrt(dx_block**2 + dy_block**2)
                            wi = (0.00708 * avg_perm_val * thickness_val) / max(np.log(max(r_o / 0.354, 1.01)), 0.1)

                        if is_inj:
                            j_peaceman_inj_base += wi
                        else:
                            j_peaceman_prod_base += wi

                current_p = initial_pressure
                p_min = _safe_float(params.get("bhp_prod"), 1000.0)
                c_o = _safe_float(params.get("compressibility_oil"), 1.0e-5)
                c_w = _safe_float(params.get("compressibility_water"), 3.0e-6)
                c_f = _safe_float(params.get("compressibility_rock"), _safe_attr(reservoir_data, "rock_compressibility", 4.0e-6))

                cum_oil_stb = 0.0
                cum_water_prod_bbl = 0.0
                cum_water_inj_bbl = 0.0
                cum_inj_mscf_running = 0.0
                cum_gas_prod_mscf_running = 0.0
                cum_caprock_leakage_tonne = 0.0
                cum_fault_leakage_tonne = 0.0

                is_fault_reactivated_flag = False
                is_caprock_breached_flag = False

                for i, t_days in enumerate(time_vector):
                    step_dt = dt[i]
                    if i == 0:
                        pressure_profile[i] = current_p
                        sat_oil_profile[i] = 1.0 - swi
                        sat_water_profile[i] = swi
                        sat_gas_profile[i] = 0.0
                        bo_profile[i] = bo_init
                        bg_profile[i] = b_co2_init
                        viscosity_oil_profile[i] = pvt_engine.calculate_oil_viscosity_cp(current_p, 0.0)
                        geomech_0 = geomech_model.evaluate_state(current_p)
                        fault_slip_profile[i] = geomech_0.slip_tendency
                        caprock_margin_profile[i] = geomech_0.caprock_safety_margin_fraction
                        vrr_profile[i] = 1.0
                        p_sandface_profile[i] = current_p
                        x_co2_profile[i] = 0.0
                        y_co2_profile[i] = 0.0
                        leakage_rate_profile[i] = 0.0
                        continue

                    # 1. Track solvent mass inventory & liquid solvent concentration x_CO2
                    cum_inj_tonne = cum_inj_mscf_running * CO2_TONNE_PER_MSCF
                    oil_mass_tonne = max(ooip * 0.135, 1.0)
                    # Solvent dissolved in oil phase (smooth saturation curve)
                    x_co2 = float(np.clip((cum_inj_tonne * 0.55) / (oil_mass_tonne + cum_inj_tonne * 0.55), 0.0, 0.85))
                    # Vapor phase solvent fraction y_co2
                    y_co2 = float(np.clip(cum_inj_mscf_running / max(cum_inj_mscf_running + 1000.0, 1.0), 0.05, 0.95))

                    # 2. Dynamic Solvent-Extended Compositional PVT
                    p_inj_sandface = min(current_p + 400.0, geomech_model.p_safe_ceiling)
                    p_prod_sandface = max(p_min, current_p - 400.0)

                    bo_dynamic = pvt_engine.calculate_oil_fvf_rb_per_stb(current_p, x_co2)
                    mu_o_dynamic = pvt_engine.calculate_oil_viscosity_cp(current_p, x_co2)
                    b_co2_inj = pvt_engine.calculate_co2_fvf_rb_per_mscf(p_inj_sandface)
                    gas_props = pvt_engine.calculate_mixture_gas_properties(current_p, y_co2)
                    bg_dynamic = gas_props["bg_rb_per_mscf"]
                    c_g_dynamic = gas_props["compressibility_psi_inv"]

                    bo_profile[i] = bo_dynamic
                    bg_profile[i] = bg_dynamic
                    viscosity_oil_profile[i] = mu_o_dynamic

                    # 3. Dynamic Volumetric Saturations & Rock Compaction
                    vp_dynamic = pore_volume_rb * (1.0 + c_f * (current_p - initial_pressure))
                    remaining_oil_stb = max(0.0, ooip - cum_oil_stb)
                    S_o = float(np.clip((remaining_oil_stb * bo_dynamic) / max(vp_dynamic, 1.0), 0.0, 1.0))
                    water_in_res_bbl = max(0.0, pore_volume_rb * swi + cum_water_inj_bbl - cum_water_prod_bbl)
                    S_w = float(np.clip(water_in_res_bbl / max(vp_dynamic, 1.0), 0.0, 1.0))
                    S_g = float(np.clip(1.0 - S_o - S_w, 0.0, 1.0))

                    sat_oil_profile[i] = S_o
                    sat_water_profile[i] = S_w
                    sat_gas_profile[i] = S_g

                    # 4. Geomechanical State, Fault Slip & Caprock Containment
                    geomech_state = geomech_model.evaluate_state(current_p)
                    fault_slip_profile[i] = geomech_state.slip_tendency
                    caprock_margin_profile[i] = geomech_state.caprock_safety_margin_fraction

                    if geomech_state.is_fault_reactivated:
                        is_fault_reactivated_flag = True
                    if geomech_state.is_caprock_breached:
                        is_caprock_breached_flag = True

                    cum_caprock_leakage_tonne += geomech_state.caprock_leakage_rate_tonne_day * step_dt
                    cum_fault_leakage_tonne += geomech_state.fault_leakage_rate_tonne_day * step_dt
                    leakage_rate_profile[i] = geomech_state.total_geomech_leakage_rate_tonne_day

                    p_sandface_profile[i] = p_inj_sandface
                    x_co2_profile[i] = x_co2
                    y_co2_profile[i] = y_co2

                    # 5. Dual-Pressure Rates & Deliverability
                    co2_inj_mscfd = profile_result["injection_profile"][i]
                    water_inj_bpd = profile_result.get("water_injection_profile", np.zeros_like(profile_result["oil_profile"]))[i]
                    q_inj_step_rb = co2_inj_mscfd * b_co2_inj + water_inj_bpd * 1.0

                    nominal_oil_stb = profile_result["oil_profile"][i]
                    nominal_water_bpd = profile_result["water_profile"][i]
                    nominal_gas_mscfd = profile_result["gas_profile"][i]

                    q_prod_step_rb = (nominal_oil_stb * bo_dynamic +
                                     nominal_water_bpd * 1.0 +
                                     nominal_gas_mscfd * bg_dynamic)

                    # Dynamic Voidage Replacement Ratio (VRR)
                    vrr_profile[i] = q_inj_step_rb / max(q_prod_step_rb, 1e-4)

                    # 6. Deliverability Coupling & Containment Constraints
                    nominal_drawdown = 500.0
                    J_inj_nominal = q_inj_step_rb / nominal_drawdown
                    J_prod_nominal = q_prod_step_rb / nominal_drawdown

                    if j_peaceman_prod_base > 0:
                        J_prod = max(j_peaceman_prod_base / max(mu_o_dynamic, 0.05), J_prod_nominal * 0.1)
                    else:
                        J_prod = J_prod_nominal

                    if j_peaceman_inj_base > 0:
                        mu_inj_eff = 0.05 if water_inj_bpd <= 0 else 0.50
                        J_inj = max(j_peaceman_inj_base / mu_inj_eff, J_inj_nominal * 0.1)
                    else:
                        J_inj = J_inj_nominal

                    prod_drawdown = max(0.0, current_p - p_min)
                    actual_q_prod = min(q_prod_step_rb, J_prod * prod_drawdown)

                    p_safe_ceiling = geomech_model.p_safe_ceiling
                    inj_drawdown_geomech = max(0.0, p_safe_ceiling - current_p)
                    q_inj_cap = J_inj * inj_drawdown_geomech

                    if current_p >= p_safe_ceiling:
                        actual_q_inj = 0.0  # EPA Class VI UIC Shut-in
                    elif current_p > target_pressure:
                        bleed_factor = max(0.0, 1.0 - (current_p - target_pressure) / 500.0)
                        actual_q_inj = min(q_inj_step_rb, q_inj_cap) * bleed_factor
                    else:
                        actual_q_inj = min(q_inj_step_rb, q_inj_cap)

                    q_net_ipr = actual_q_inj - actual_q_prod

                    # 7. Material Balance ODE Pressure Increment
                    ct_dynamic = max(c_o * S_o + c_g_dynamic * S_g + c_w * S_w + c_f, 1e-6)
                    J_eff = max(J_inj + J_prod, 1e-5)
                    dp = (q_net_ipr * step_dt) / (vp_dynamic * ct_dynamic + J_eff * step_dt)
                    dp = float(np.clip(dp, -450.0, 450.0))

                    tolerance_window = 50.0
                    k_smooth = 1.0 / (tolerance_window * 0.33)
                    if dp > 0:
                        headroom = max(0.0, p_safe_ceiling - current_p)
                        dp *= max(0.0, 1.0 - np.exp(-k_smooth * headroom))
                    elif dp < 0:
                        margin = max(0.0, current_p - p_min)
                        dp *= max(0.0, 1.0 - np.exp(-k_smooth * margin))

                    current_p += dp
                    current_p = float(np.clip(current_p, p_min, p_safe_ceiling))
                    pressure_profile[i] = current_p

                    # Cumulatives update
                    cum_oil_stb += nominal_oil_stb * step_dt
                    cum_water_prod_bbl += nominal_water_bpd * step_dt
                    cum_water_inj_bbl += water_inj_bpd * step_dt
                    cum_inj_mscf_running += co2_inj_mscfd * step_dt
                    cum_gas_prod_mscf_running += nominal_gas_mscfd * step_dt
            else:
                pressure_profile = np.array([initial_pressure])
                sat_oil_profile = np.array([1.0 - 0.25])
                sat_gas_profile = np.array([0.0])
                sat_water_profile = np.array([0.25])
                bo_profile = np.array([1.2])
                bg_profile = np.array([0.5])
                viscosity_oil_profile = np.array([2.0])
                fault_slip_profile = np.array([0.3])
                caprock_margin_profile = np.array([1.0])
                vrr_profile = np.array([1.0])

            # Recouple simulated dynamic mean pressure and throughput with PhD recovery model
            mean_p = float(np.mean(pressure_profile))
            dt_days = np.diff(time_vector, prepend=0)
            cum_inj_mscf = float(np.sum(profile_result["injection_profile"] * dt_days))
            b_co2_mean = pvt_engine.calculate_co2_fvf_rb_per_mscf(mean_p)
            cum_inj_rb = cum_inj_mscf * b_co2_mean
            ooip_val = float(getattr(reservoir_data, "ooip_stb", 1e6))
            swi_val = float(getattr(reservoir_data, "initial_water_saturation", getattr(reservoir_data, "connate_water_saturation", 0.25)))
            simulated_hcpvi = float(cum_inj_rb / max(ooip_val * b_co2_mean / max(1.0 - swi_val, 0.05), 1e-6))

            rec_params = params.copy()
            rec_params["pressure"] = mean_p
            rec_params["hcpvi"] = simulated_hcpvi
            rec_params["x_co2"] = float(np.clip(cum_inj_mscf * CO2_TONNE_PER_MSCF / max(ooip_val * 0.135, 1.0), 0.0, 0.85))

            if hasattr(self.surrogate_model, "calculate_recovery"):
                recovery_factor = float(self.surrogate_model.calculate_recovery(**rec_params))
            elif hasattr(self.surrogate_model, "recovery_model") and hasattr(self.surrogate_model.recovery_model, "calculate_recovery"):
                recovery_factor = float(self.surrogate_model.recovery_model.calculate_recovery(**rec_params))

            rf_max_physical = max(0.0, 1.0 - swi_val - getattr(reservoir_data, "residual_oil_saturation", params.get("sor", 0.25)))
            recovery_factor = float(np.clip(recovery_factor, 0.0, rf_max_physical))

            # Scale profile_result["oil_profile"] so cumulative production equals recovery_factor * ooip
            oil_profile = profile_result["oil_profile"]
            cum_oil_shape = np.cumsum(oil_profile * dt_days)
            max_cum_shape = cum_oil_shape[-1] if len(cum_oil_shape) > 0 and cum_oil_shape[-1] > 0 else 1.0

            target_cum_oil = recovery_factor * ooip_val
            if max_cum_shape > 0 and target_cum_oil > 0:
                profile_result["oil_profile"] = oil_profile * (target_cum_oil / max_cum_shape)
                rf_profile = np.cumsum(profile_result["oil_profile"] * dt_days) / max(ooip_val, 1.0)
            else:
                rf_profile = cum_oil_shape / max(ooip_val, 1.0)

            # Extract 4-stream profiles
            oil_rate = profile_result["oil_profile"]
            water_rate = profile_result["water_profile"]
            total_gas_rate = profile_result["gas_profile"]
            co2_prod_rate = profile_result.get("co2_gas_profile", profile_result["gas_profile"])
            hc_gas_rate = profile_result.get("solution_gas_profile", np.maximum(0.0, total_gas_rate - co2_prod_rate))
            inj_rate = profile_result["injection_profile"]
            water_inj_rate = profile_result.get("water_injection_profile", np.zeros_like(oil_rate))

            cum_oil_total = float(np.sum(oil_rate * dt_days))
            cum_water_prod_bbl = float(np.sum(water_rate * dt_days))
            cum_water_inj_bbl = float(np.sum(water_inj_rate * dt_days))
            cum_co2_prod_mscf = float(np.sum(co2_prod_rate * dt_days))
            cum_hc_gas_mscf = float(np.sum(hc_gas_rate * dt_days))
            cum_total_gas_mscf = float(np.sum(total_gas_rate * dt_days))

            cum_co2_inj_tonne = cum_inj_mscf * CO2_TONNE_PER_MSCF
            cum_co2_prod_tonne = cum_co2_prod_mscf * CO2_TONNE_PER_MSCF

            cum_stored_mscf = max(0.0, cum_inj_mscf - cum_co2_prod_mscf)
            co2_stored_tonnes = cum_stored_mscf * CO2_TONNE_PER_MSCF

            net_utilization = cum_stored_mscf / max(cum_oil_total, 1.0)
            gross_utilization = cum_inj_mscf / max(cum_oil_total, 1.0)

            # Breakthrough time estimation
            bt_years = float(params.get("breakthrough_time_years", profile_result.get("breakthrough_time_years", 1.0)))
            if bt_years <= 0 or bt_years > operational_params.project_lifetime_years:
                gas_threshold = 0.05 * np.max(co2_prod_rate) if len(co2_prod_rate) > 0 and np.max(co2_prod_rate) > 0 else 1.0
                bt_indices = np.where(co2_prod_rate > gas_threshold)[0]
                bt_years = float(time_vector[bt_indices[0]] / 365.25) if len(bt_indices) > 0 else float(operational_params.project_lifetime_years)

            omega_val = float(getattr(self.surrogate_model, "last_omega", getattr(getattr(self.surrogate_model, "recovery_model", None), "last_omega", 0.0)))
            storage_eff = float(prediction.get("storage_efficiency", cum_stored_mscf / max(cum_inj_mscf, 1.0)))

            # Calculate annual profiles and enforce compressor bottlenecks & availability
            n_years = int(operational_params.project_lifetime_years)
            annual_oil_stb = np.zeros(n_years)
            annual_water_prod_bbl = np.zeros(n_years)
            annual_water_inj_bbl = np.zeros(n_years)
            annual_hc_gas_mscf = np.zeros(n_years)
            annual_co2_prod_mscf = np.zeros(n_years)
            annual_co2_inj_mscf = np.zeros(n_years)
            annual_fault_leakage_tonne = np.zeros(n_years)
            annual_caprock_leakage_tonne = np.zeros(n_years)

            q_comp_max_annual_mscf = _safe_attr(eor_params, "recycle_compressor_capacity_mscfd", 50000.0) * 365.25 * _safe_attr(eor_params, "facility_availability", 0.95)
            recycle_loss_frac = _safe_attr(eor_params, "co2_recycle_loss_fraction", 0.05)
            co2_recycle_eff = min(_safe_float(params.get("co2_recycling_efficiency"), 0.95), 1.0 - recycle_loss_frac)

            if len(time_vector) > 1:
                t_mids = 0.5 * (time_vector[:-1] + time_vector[1:])
                dts = np.diff(time_vector)
                for j, (t_mid, dt_step) in enumerate(zip(t_mids, dts)):
                    y = min(n_years - 1, max(0, int(t_mid // 365.25)))
                    annual_oil_stb[y] += float(oil_rate[j + 1] * dt_step)
                    annual_water_prod_bbl[y] += float(water_rate[j + 1] * dt_step)
                    annual_water_inj_bbl[y] += float(water_inj_rate[j + 1] * dt_step)
                    annual_hc_gas_mscf[y] += float(hc_gas_rate[j + 1] * dt_step)
                    annual_co2_prod_mscf[y] += float(co2_prod_rate[j + 1] * dt_step)
                    annual_co2_inj_mscf[y] += float(inj_rate[j + 1] * dt_step)
            else:
                annual_oil_stb[0] = float(np.sum(oil_rate) * 365.25)
                annual_water_prod_bbl[0] = float(np.sum(water_rate) * 365.25)
                annual_co2_prod_mscf[0] = float(np.sum(co2_prod_rate) * 365.25)
                annual_co2_inj_mscf[0] = float(np.sum(inj_rate) * 365.25)

            # Recycle is strictly capped by compressor bottleneck and available injection demand
            annual_co2_recycled_mscf = np.minimum(
                annual_co2_prod_mscf * co2_recycle_eff,
                np.minimum(q_comp_max_annual_mscf, annual_co2_inj_mscf)
            )
            annual_co2_purchased_mscf = np.maximum(0.0, annual_co2_inj_mscf - annual_co2_recycled_mscf)

            # Ratios
            water_cut_profile = water_rate / np.maximum(water_rate + oil_rate, 1e-6)
            total_gor_profile = (total_gas_rate * 1000.0) / np.maximum(oil_rate, 1e-4)
            hc_gor_profile = (hc_gas_rate * 1000.0) / np.maximum(oil_rate, 1e-4)
            wag_ratio_inst = np.where(inj_rate > 0, water_inj_rate / np.maximum(inj_rate, 1e-4), 0.0)
            cum_vol_wag_ratio = cum_water_inj_bbl / max(cum_inj_mscf * b_co2_init, 1e-4)

            # Daily rates and profiles
            q_comp_max_daily_mscfd = _safe_attr(eor_params, "recycle_compressor_capacity_mscfd", 50000.0) * _safe_attr(eor_params, "facility_availability", 0.95)
            co2_recycled_rate = np.minimum(co2_prod_rate * co2_recycle_eff, np.minimum(q_comp_max_daily_mscfd, inj_rate))
            co2_purchased_rate = np.maximum(0.0, inj_rate - co2_recycled_rate)
            cum_oil_profile = np.cumsum(oil_rate * dt_days)
            cum_hc_gas_profile = np.cumsum(hc_gas_rate * dt_days)
            cum_water_profile = np.cumsum(water_rate * dt_days)
            cum_co2_inj_profile = np.cumsum(inj_rate * dt_days)
            cum_co2_purchased_profile = np.cumsum(co2_purchased_rate * dt_days)
            cum_co2_recycled_profile = np.cumsum(co2_recycled_rate * dt_days)
            cum_co2_prod_profile = np.cumsum(co2_prod_rate * dt_days)
            monthly_oil_stb = profile_result.get("monthly_oil_stb", np.zeros(1))

            profiles_dict = {
                "time_vector": profile_result["time_vector"],
                # Stream 1: Crude Oil
                "oil_profile": oil_rate,
                "oil_production_rate": oil_rate,
                "cumulative_oil_bbl": cum_oil_profile,
                "cumulative_oil_stb": cum_oil_profile,
                "annual_oil_stb": annual_oil_stb,
                "yearly_oil_stb": annual_oil_stb,
                "monthly_oil_stb": monthly_oil_stb,
                "oil_fvf_profile": bo_profile,
                # Stream 2: Natural Gas
                "hydrocarbon_gas_sales_mscfd": hc_gas_rate,
                "hydrocarbon_gas_production_rate": hc_gas_rate,
                "solution_gas_profile": hc_gas_rate,
                "annual_hydrocarbon_gas_sales_mscf": annual_hc_gas_mscf,
                "yearly_hydrocarbon_gas_mscf": annual_hc_gas_mscf,
                "cumulative_hydrocarbon_gas_sales_mscf": cum_hc_gas_profile,
                "gas_profile": total_gas_rate,
                "total_gas_production_rate": total_gas_rate,
                # Stream 3: Water
                "water_profile": water_rate,
                "water_production_rate": water_rate,
                "water_cut_profile": water_cut_profile,
                "water_cut": water_cut_profile,
                "cumulative_water_bbl": cum_water_profile,
                "annual_water_bbl": annual_water_prod_bbl,
                # Stream 4: Injection Agent
                "injection_profile": inj_rate,
                "co2_injection": inj_rate,
                "co2_purchased_mscfd": co2_purchased_rate,
                "co2_recycled_mscfd": co2_recycled_rate,
                "co2_gas_profile": co2_prod_rate,
                "co2_production_rate": co2_prod_rate,
                "water_injection_profile": water_inj_rate,
                "cumulative_co2_injected_mscf": cum_co2_inj_profile,
                "cumulative_co2_purchased_mscf": cum_co2_purchased_profile,
                "cumulative_co2_recycled_mscf": cum_co2_recycled_profile,
                "annual_co2_purchased_mscf": annual_co2_purchased_mscf,
                "yearly_co2_purchased_mscf": annual_co2_purchased_mscf,
                "annual_co2_recycled_mscf": annual_co2_recycled_mscf,
                "yearly_co2_recycled_mscf": annual_co2_recycled_mscf,
                "annual_co2_injected_mscf": annual_co2_inj_mscf,
                "yearly_co2_injected_mscf": annual_co2_inj_mscf,
                # State Tracks
                "pressure": pressure_profile,
                "reservoir_pressure": pressure_profile,
                "sandface_injection_pressure": p_sandface_profile,
                "vrr_local": vrr_profile,
                "fault_slip_tendency": fault_slip_profile,
                "caprock_tensile_margin": caprock_margin_profile,
                "caprock_shear_margin": caprock_margin_profile,
                "leakage_rate_tonnes_day": leakage_rate_profile,
                "x_co2_liquid": x_co2_profile,
                "y_co2_vapor": y_co2_profile,
                "saturation_oil": sat_oil_profile,
                "saturation_water": sat_water_profile,
                "saturation_gas": sat_gas_profile,
            }

            total_leakage_tonne = cum_caprock_leakage_tonne + cum_fault_leakage_tonne

            return {
                "profiles": profiles_dict,
                # 1. Crude Oil Stream
                "oil_production_rate": oil_rate,
                "oil_profile": oil_rate,
                "cumulative_oil": cum_oil_total,
                "cumulative_oil_stb": cum_oil_total,
                "cumulative_oil_bbl": cum_oil_profile,
                "annual_oil_stb": annual_oil_stb,
                "yearly_oil_stb": annual_oil_stb,
                "recovery_factor": recovery_factor,
                "recovery_factor_profile": rf_profile,
                "npv": npv,

                # 2. Hydrocarbon Natural Gas Stream
                "hydrocarbon_gas_production_rate": hc_gas_rate,
                "solution_gas_profile": hc_gas_rate,
                "cumulative_hydrocarbon_gas_mscf": cum_hc_gas_mscf,
                "annual_hydrocarbon_gas_mscf": annual_hc_gas_mscf,
                "yearly_hydrocarbon_gas_mscf": annual_hc_gas_mscf,
                "hydrocarbon_gor_scf_per_stb": float(np.mean(hc_gor_profile)),

                # 3. Water Stream
                "water_production_rate": water_rate,
                "water_profile": water_rate,
                "cumulative_water_produced_bbl": cum_water_prod_bbl,
                "annual_water_produced_bbl": annual_water_prod_bbl,
                "yearly_water_produced_bbl": annual_water_prod_bbl,
                "water_cut": water_cut_profile,
                "water_cut_profile": water_cut_profile,

                # 4. Injection Agent Stream (CO2 + WAG Water)
                "co2_injection": inj_rate,
                "co2_injection_rate": inj_rate,
                "injection_profile": inj_rate,
                "cumulative_co2_injected_mscf": cum_inj_mscf,
                "cumulative_co2_injected_tonne": cum_co2_inj_tonne,
                "annual_co2_injected_mscf": annual_co2_inj_mscf,
                "yearly_co2_injected_mscf": annual_co2_inj_mscf,
                "annual_co2_purchased_mscf": annual_co2_purchased_mscf,
                "yearly_co2_purchased_mscf": annual_co2_purchased_mscf,
                "annual_co2_recycled_mscf": annual_co2_recycled_mscf,
                "yearly_co2_recycled_mscf": annual_co2_recycled_mscf,
                "cumulative_co2_purchased_mscf": float(np.sum(annual_co2_purchased_mscf)),
                "cumulative_co2_recycled_mscf": float(np.sum(annual_co2_recycled_mscf)),
                "co2_production_rate": co2_prod_rate,
                "co2_gas_profile": co2_prod_rate,
                "co2_production_cumulative": np.cumsum(co2_prod_rate * dt_days),
                "cumulative_co2_produced_mscf": cum_co2_prod_mscf,
                "cumulative_co2_produced_tonne": cum_co2_prod_tonne,
                "annual_co2_produced_mscf": annual_co2_prod_mscf,
                "yearly_co2_produced_mscf": annual_co2_prod_mscf,
                "cumulative_co2_stored_mscf": cum_stored_mscf,
                "co2_stored": co2_stored_tonnes,
                "co2_stored_tonnes": co2_stored_tonnes,
                "storage_efficiency": storage_eff,
                "net_utilization_mscf_per_stb": net_utilization,
                "gross_utilization_mscf_per_stb": gross_utilization,
                "water_injection_rate": water_inj_rate,
                "water_injection_profile": water_inj_rate,
                "cumulative_water_injected_bbl": cum_water_inj_bbl,
                "annual_water_injected_bbl": annual_water_inj_bbl,
                "yearly_water_injected_bbl": annual_water_inj_bbl,
                "instantaneous_wag_ratio": wag_ratio_inst,
                "cumulative_volumetric_wag_ratio": float(cum_vol_wag_ratio),

                # 5. Total Gas & Voidage Balances
                "gas_production_rate": total_gas_rate,
                "total_gas_production_rate": total_gas_rate,
                "gas_profile": total_gas_rate,
                "cumulative_total_gas_mscf": cum_total_gas_mscf,
                "producing_gor_scf_per_stb": float(np.mean(total_gor_profile)),
                "voidage_replacement_ratio": vrr_profile,
                "voidage_replacement_ratio_profile": vrr_profile,

                # 6. Pressure, PVT & Geomechanical In-Situ State
                "pressure": pressure_profile,
                "pressure_profile": pressure_profile,
                "pressure_model_based": True,
                "mean_pressure_psi": float(np.mean(pressure_profile)),
                "max_pressure_psi": float(np.max(pressure_profile)),
                "min_pressure_psi": float(np.min(pressure_profile)),
                "saturation_oil_profile": sat_oil_profile,
                "saturation_gas_profile": sat_gas_profile,
                "saturation_water_profile": sat_water_profile,
                "bo_profile": bo_profile,
                "bg_profile": bg_profile,
                "viscosity_oil_profile": viscosity_oil_profile,
                "fault_slip_tendency_profile": fault_slip_profile,
                "caprock_safety_margin_profile": caprock_margin_profile,
                "is_fault_reactivated": bool(is_fault_reactivated_flag),
                "is_caprock_breached": bool(is_caprock_breached_flag),
                "total_leakage_tonne": float(total_leakage_tonne),
                "annual_fault_leakage_tonne": annual_fault_leakage_tonne,
                "annual_caprock_leakage_tonne": annual_caprock_leakage_tonne,
                "breakthrough_time_years": bt_years,
                "miscibility_degree": omega_val,
                "average_miscibility_degree": omega_val,
                "time_vector": profile_result["time_vector"],

                # Metadata & Integration
                "simulation_time": evaluation_time,
                "engine_type": "surrogate",
                "convergence_status": "success",
                "constraint_violations": {},
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
            "WAG_ratio": getattr(eor_params, "wag_ratio", getattr(eor_params, "WAG_ratio", 1.0)),
            "wag_ratio": getattr(eor_params, "wag_ratio", getattr(eor_params, "WAG_ratio", 1.0)),
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
            # Default bg is 0.0005 RB/SCF (0.5 RB/MSCF) -> 1 / (0.0005 * 1000) = 2.0 MSCF/RB
            "mscf_per_res_bbl": 1.0 / (max(getattr(reservoir_data, "bg", 0.0005), 1e-6) * 1000.0),

            # Simulation parameters
            "project_lifetime_years": operational_params.project_lifetime_years,

            # Empirical fitting parameters for surrogate model calibration
            "c7_plus": self.fitting_params.c7_plus_fraction,
            "alpha_base": self.fitting_params.alpha_base,
            "miscibility_window": self.fitting_params.miscibility_window,
            "breakthrough_time": getattr(self.fitting_params, "breakthrough_time_years", getattr(self.fitting_params, "breakthrough_time", None)),
            "trapping_efficiency": self.fitting_params.trapping_efficiency,
            "initial_gor": self.fitting_params.initial_gor_scf_per_stb,
            "transverse_mixing_calibration": self.fitting_params.transverse_mixing_calibration,
            "omega_tl": self.fitting_params.omega_tl,
            "k_ro_0": self.fitting_params.k_ro_0,
            "k_rg_0": self.fitting_params.k_rg_0,
            "n_o": self.fitting_params.n_o,
            "n_g": self.fitting_params.n_g,
        }

        # Calculate dynamic HCPVI: (Inj_rate_res_bbl/d * 365.25 * years) / HydrocarbonPoreVolume_res_bbl
        ooip = reservoir_data.ooip_stb or 1e6
        swi = getattr(reservoir_data, "initial_water_saturation", 0.25)
        bo = 1.2 # Formation volume factor
        pv_rb = (ooip * bo) / max(1.0 - swi, 0.1)
        
        # injection_rate is MSCFD; mscf_per_res_bbl is MSCF/RB
        # q_inj_rb_day = injection_rate [MSCFD] / mscf_per_res_bbl [MSCF/RB] = RB/day
        mscf_per_rb_hcpvi = params.get("mscf_per_res_bbl", 2.0)  # MSCF/RB (will be overwritten after params built)
        q_inj_rb_day_hcpvi = eor_params.injection_rate / max(mscf_per_rb_hcpvi, 1e-6)
        total_inj_rb = q_inj_rb_day_hcpvi * 365.25 * operational_params.project_lifetime_years
        params["hcpvi"] = total_inj_rb / max(pv_rb, 1.0)

        # First-Principles Breakthrough Time from Koval (1963) Fractional Flow
        if params.get("breakthrough_time") is None:
            v_dp = float(params.get("v_dp", 0.5))
            H_k = 10.0 ** (v_dp / max(1.0 - v_dp, 1e-4))
            M_eff = float(params.get("mobility_ratio", 2.5))
            E_eff = (0.78 + 0.22 * (M_eff**0.25))**4
            K_koval = max(1.01, H_k * E_eff)
            
            # Use geometric pore volume if available (area in acres, thickness in ft, porosity):
            area_acres = float(getattr(reservoir_data, "area_acres", 0.0) or 0.0)
            thickness_ft = float(getattr(reservoir_data, "thickness_ft", 0.0) or 0.0)
            porosity = float(getattr(reservoir_data, "average_porosity", 0.0) or 0.0)
            if area_acres > 0 and thickness_ft > 0 and porosity > 0:
                pv_calc = (area_acres * 43560.0 * thickness_ft * porosity) / 5.615
            else:
                pv_calc = pv_rb
            
            # Injection rate in RB/day:
            bg_val = float(getattr(reservoir_data, "bg_rb_per_mscf", params.get("bg_rb_per_mscf", 0.5)))
            q_inj_rb_day = eor_params.injection_rate * bg_val
            
            # WAG mobility buffering delays breakthrough
            inj_scheme = str(params.get("injection_scheme", "continuous")).lower()
            wag_factor = (1.0 + float(params.get("wag_ratio", 1.0))) if inj_scheme in ["wag", "swag"] else 1.0
            
            # Breakthrough time in years: (PoreVolume / K_koval) / (annual injection volume) * wag_factor
            annual_inj_rb = max(q_inj_rb_day * 365.25, 1.0)
            t_bt_calc = (pv_calc / K_koval) / annual_inj_rb * wag_factor
            
            # Bound within physical lifetime
            lifetime = float(operational_params.project_lifetime_years)
            bt_time = float(np.clip(t_bt_calc, 0.1, max(lifetime * 0.9, 0.5)))
            params["breakthrough_time"] = bt_time
            params["breakthrough_time_years"] = bt_time

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

        # Add economic parameters — explicitly maps EconomicParameters dataclass fields
        if economic_params:
            params.update({
                "oil_price_usd_per_bbl": economic_params.oil_price_usd_per_bbl,
                "co2_cost_usd_per_ton": economic_params.co2_purchase_cost_usd_per_tonne,
                "co2_purchase_cost_usd_per_tonne": economic_params.co2_purchase_cost_usd_per_tonne,
                "co2_recycle_cost_usd_per_tonne": economic_params.co2_recycle_cost_usd_per_tonne,
                "discount_rate": economic_params.discount_rate_fraction,
                "discount_rate_fraction": economic_params.discount_rate_fraction,
                # CAPEX: was silently ignored due to attribute name mismatch
                "capex_usd": getattr(economic_params, "capex_usd", 0.0),
                # Variable OPEX per barrel: was silently ignored (stale attr name)
                "variable_opex_usd_per_bbl": getattr(economic_params, "variable_opex_usd_per_bbl", 5.0),
                # CO2 storage credit: was silently ignored (looked in wrong object)
                "co2_storage_credit_usd_per_tonne": getattr(economic_params, "co2_storage_credit_usd_per_tonne", 0.0),
                # Carbon tax on leakage (economic_params takes precedence over advanced_engine_params)
                "carbon_tax_usd_per_tonne": getattr(economic_params, "carbon_tax_usd_per_tonne", 0.0),
            })

        # Well deliverability: productivity_index from eor_params (was always defaulting to 5.0)
        params["productivity_index"] = getattr(eor_params, "productivity_index", 5.0)

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
