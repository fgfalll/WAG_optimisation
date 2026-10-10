"""
Material balance module for CO₂ EOR and storage analysis.
Handles CO₂ accounting, breakthrough physics, recycling calculations, and generates material balance graphs.
Designed to be separate from the optimization engine to avoid performance impact.
"""

import numpy as np

import plotly.graph_objects as go
from plotly.subplots import make_subplots


from typing import Dict, List, Optional, Tuple, Any
import logging

logger = logging.getLogger(__name__)

# Import breakthrough physics
try:
    from .breakthrough_physics import CO2BreakthroughPhysics, BreakthroughParameters
except ImportError:
    logger.warning("Breakthrough physics module not available - using fallback calculations")

    # Fallback definitions for testing
    class BreakthroughParameters:
        breakthrough_gor_threshold = 800.0
        recycling_efficiency = 0.9

    class CO2BreakthroughPhysics:
        def __init__(self, params=None):
            self.params = params or BreakthroughParameters()


class MaterialBalanceAnalyzer:
    """
    Analyzes CO₂ material balance for EOR projects with storage verification.
    Calculates injection, production, recycling, and net storage metrics.
    Generates comprehensive material balance graphs.
    """

    def __init__(
        self,
        co2_density_tonne_per_mscf: float = 0.053,
        breakthrough_params: Optional[BreakthroughParameters] = None,
        eos_model: Optional[Any] = None,
    ):
        """
        Initialize the material balance analyzer with breakthrough physics.

        Args:
            co2_density_tonne_per_mscf: Density conversion factor from Mscf to tonnes
            breakthrough_params: Parameters for breakthrough physics calculations
            eos_model: Optional ReservoirFluid EOS model for dynamic property calculation
        """
        self.co2_density_tonne_per_mscf = co2_density_tonne_per_mscf
        self.eos_model = eos_model
        self.breakthrough_physics = CO2BreakthroughPhysics(breakthrough_params)
        self._eos_warning_logged = False

    def _get_co2_fraction_at_reservoir(
        self, reservoir_params: Optional[Dict], eor_params: Optional[Dict]
    ) -> float:
        """
        Get CO2 mole fraction at reservoir conditions from EOS model.

        Args:
            reservoir_params: Reservoir properties (pressure, temperature)
            eor_params: EOR operation parameters

        Returns:
            CO2 mole fraction (0.0-1.0)
        """
        if self.eos_model is not None and reservoir_params is not None:
            try:
                pressure = reservoir_params.get("pressure", 2000.0)
                temp_f = reservoir_params.get("temperature", 150.0)
                temp_k = (temp_f - 32) * 5.0 / 9.0 + 273.15
                pres_pa = pressure * 6894.76
                props = self.eos_model.get_properties_si(temp_k, pres_pa)
                phase = props.get("phase", "V")
                if phase == "V":
                    z_vapor = props.get("vapor_properties", {}).get("Z", 1.0)
                    z_liquid = props.get("liquid_properties", {}).get("Z", 1.0)
                    avg_z = (z_vapor + z_liquid) / 2.0
                    vapor_frac = min(1.0, max(0.0, 1.0 - avg_z + 0.2))
                else:
                    vapor_frac = 0.05

                co2_frac = 0.8
                params = getattr(self.eos_model, "eos_model", None)
                if params is None:
                    params = getattr(self.eos_model, "params", None)
                if params is not None:
                    if hasattr(params, "mole_fractions"):
                        mf = np.asarray(params.mole_fractions)
                        if len(mf) > 0:
                            co2_frac = float(mf[0])
                    elif hasattr(params, "component_properties"):
                        cp = np.asarray(params.component_properties)
                        if cp.ndim == 2 and cp.shape[0] > 0:
                            co2_frac = float(cp[0, 0])

                co2_frac_in_vapor = vapor_frac
                co2_frac_in_liquid = 1.0 - vapor_frac
                co2_fraction = (
                    co2_frac * co2_frac_in_vapor
                    + co2_frac * (1.0 - co2_frac_in_vapor) * co2_frac_in_liquid
                )
                co2_fraction = min(0.95, max(0.05, co2_fraction))
                return float(co2_fraction)
            except Exception as e:
                if not self._eos_warning_logged:
                    logger.warning(
                        f"EOS-based CO2 fraction calculation failed, using fallback: {e}"
                    )
                    self._eos_warning_logged = True
        return 0.50

    def calculate_material_balance(
        self,
        annual_co2_injected_mscf: np.ndarray,
        annual_co2_produced_mscf: np.ndarray,
        annual_co2_recycled_mscf: np.ndarray,
        leakage_rate_fraction: float = 0.01,
        reservoir_params: Optional[Dict] = None,
        eor_params: Optional[Dict] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Calculate comprehensive CO₂ material balance with leakage accounting.

        Uses OSTI-1204577 (Peck et al. 2017) methodology:
        - CO2 stored = CO2 purchased - CO2 produced  [Equation 6]
        - CO2 produced includes recycled portion that returns to surface
        - Net injection = purchased - recycled (recycled CO2 is not "new" injection)
        - Only purchased CO2 counts toward storage efficiency

        Args:
            annual_co2_injected_mscf: Annual CO₂ injected (Mscf) - should be CO2_purchased
            annual_co2_produced_mscf: Annual CO₂ produced (Mscf) - total produced gas
            annual_co2_recycled_mscf: Annual CO₂ recycled (Mscf)
            leakage_rate_fraction: Annual leakage rate as fraction of cumulative storage
            reservoir_params: Reservoir properties for breakthrough physics
            eor_params: EOR operation parameters for breakthrough physics

        Returns:
            Dictionary containing all material balance components in tonnes
        """
        # CO2 density conversion
        co2_density = self.co2_density_tonne_per_mscf

        # NOTE: annual_co2_injected_mscf is actually CO2_purchased per OSTI-1204577
        # Net injection = purchased - recycled (Equation 6: CO2 stored = purchased - produced)
        # The recycled portion of produced CO2 is NOT additional purchase
        purchased_tonne = annual_co2_injected_mscf * co2_density
        produced_tonne_raw = annual_co2_produced_mscf * co2_density
        recycled_tonne_raw = annual_co2_recycled_mscf * co2_density

        # Calculate breakthrough-aware recycling only if engine did not supply recycled CO2
        if (
            np.all(recycled_tonne_raw <= 0)
            and reservoir_params is not None
            and eor_params is not None
        ):
            breakthrough_analysis = self.calculate_breakthrough_aware_recycling(
                purchased_tonne, reservoir_params, eor_params
            )
            if breakthrough_analysis:
                recycled_tonne_raw = breakthrough_analysis.get(
                    "recycled_tonne_breakthrough_aware", recycled_tonne_raw
                )

        # The stream annual_co2_produced_mscf is already pure CO2 produced by the EOR operation
        # (calculated by FastProfileGenerator and SurrogateEngine).
        # We preserve full mass conservation without artificial shrinkage.
        produced_co2_tonne = produced_tonne_raw
        produced_hc_tonne = 0.0

        # Net injection entering system boundary = purchased CO2
        # Reference: OSTI-1204577 Equation 6: CO2 stored = CO2 purchased - uncaptured CO2 produced
        # Recycled CO2 is returned to reservoir; uncaptured portion is lost at surface.
        # Recycled amount cannot exceed produced CO2.
        recycled_tonne = np.minimum(recycled_tonne_raw, produced_co2_tonne)
        uncaptured_produced_co2 = np.maximum(0.0, produced_co2_tonne - recycled_tonne)
        net_injection_tonne = purchased_tonne

        n_years = len(purchased_tonne)

        # Geomechanical check for leakage gating:
        # Intact caprock below 90% Pfrac exhibits negligible matrix leakage (permeation <= 1e-4 / yr).
        # Subsurface leakage occurs primarily if reservoir pressure exceeds safe fracture ceiling (0.90 * Pfrac)
        # or if explicit fault leakage is configured.
        p_res = 2000.0
        if reservoir_params:
            p_res = float(reservoir_params.get("pressure", 2000.0))

        p_frac = 5500.0
        safety_factor = 0.90
        has_fault = False
        if eor_params:
            p_frac = float(eor_params.get("caprock_fracture_pressure_psi", 5500.0))
            safety_factor = float(eor_params.get("caprock_safety_factor", 0.90))
            has_fault = bool(eor_params.get("fault_leakage_enabled", False))

        p_safe_ceiling = p_frac * safety_factor
        if not has_fault and p_res <= p_safe_ceiling:
            # Mechanically intact reservoir: matrix leakage rate bounded to trace/zero
            effective_leakage_fraction = min(leakage_rate_fraction, 0.0001) if leakage_rate_fraction > 0 else 0.0
        else:
            # Caprock fracture limit exceeded or fault present: apply full leakage fraction
            effective_leakage_fraction = leakage_rate_fraction

        # Initialize arrays
        net_stored_tonne = np.zeros(n_years)
        cumulative_stored_tonne = np.zeros(n_years)
        annual_leakage_tonne = np.zeros(n_years)
        storage_efficiency = np.zeros(n_years)

        # Calculate material balance with leakage
        for i in range(n_years):
            # Net stored before leakage = purchased CO2 - uncaptured produced CO2
            # (Recycled CO2 re-enters reservoir and cancels out from net mass balance)
            net_stored_before_leakage = purchased_tonne[i] - uncaptured_produced_co2[i]

            # Calculate leakage from previous cumulative storage
            if i == 0:
                prev_stored = 0.0
            else:
                prev_stored = max(0.0, cumulative_stored_tonne[i - 1])

            # Leakage cannot exceed available stored CO2 and cannot cause negative net stored
            # Physical constraint: cannot leak more than exists in reservoir
            max_leakage = min(
                prev_stored * effective_leakage_fraction, max(0.0, net_stored_before_leakage)
            )
            leakage = max_leakage

            annual_leakage_tonne[i] = leakage

            # Net stored = net stored before leakage - leakage
            net_stored = max(0.0, net_stored_before_leakage - leakage)
            net_stored_tonne[i] = net_stored

            # Update cumulative storage
            if i == 0:
                cumulative_stored_tonne[i] = net_stored
            else:
                cumulative_stored_tonne[i] = max(0.0, cumulative_stored_tonne[i - 1]) + net_stored

            # Calculate storage efficiency = net stored / purchased
            # OSTI-1204577 efficiency = CO2 stored / CO2 purchased
            if purchased_tonne[i] > 0:
                storage_efficiency[i] = float(
                    np.clip(net_stored_tonne[i] / purchased_tonne[i], 0.0, 1.0)
                )
            else:
                storage_efficiency[i] = 0.0

        # Calculate mass balance error: total purchased vs accounted (stored + leakage + uncaptured)
        total_purchased = np.sum(purchased_tonne)
        total_accounted = (
            np.sum(net_stored_tonne)
            + np.sum(annual_leakage_tonne)
            + np.sum(uncaptured_produced_co2)
        )

        if total_purchased > 0:
            mass_balance_error = abs(total_accounted - total_purchased) / total_purchased
        else:
            mass_balance_error = 0.0

        return {
            "injected_tonne": purchased_tonne,
            "produced_co2_tonne": produced_co2_tonne,
            "produced_hc_tonne": produced_hc_tonne,
            "produced_tonne": produced_co2_tonne,
            "recycled_tonne": recycled_tonne,
            "net_injected_tonne": net_injection_tonne,
            "net_stored_tonne": net_stored_tonne,
            "cumulative_stored_tonne": cumulative_stored_tonne,
            "annual_leakage_tonne": annual_leakage_tonne,
            "storage_efficiency": storage_efficiency,
            "years": np.arange(1, n_years + 1),
            "mass_balance_error": mass_balance_error,
            "total_injected_tonne": total_purchased,
            "total_accounted_tonne": total_accounted,
        }

    def calculate_breakthrough_aware_recycling(
        self, injected_tonne: np.ndarray, reservoir_params: Dict, eor_params: Dict
    ) -> Optional[Dict[str, np.ndarray]]:
        """
        Calculate recycling considering breakthrough physics and timing.

        Args:
            injected_tonne: Annual CO₂ injected (tonnes)
            reservoir_params: Reservoir properties
            eor_params: EOR operation parameters

        Returns:
            Dictionary with breakthrough-aware recycling metrics
        """
        try:
            # Calculate breakthrough time
            breakthrough_time = self.breakthrough_physics.calculate_breakthrough_time(
                reservoir_params, eor_params, eos_model=self.eos_model
            )
            # Ensure breakthrough_time is a scalar to avoid broadcast errors
            if hasattr(breakthrough_time, "item"):
                breakthrough_time = breakthrough_time.item()
            breakthrough_time = float(breakthrough_time)

            n_years = len(injected_tonne)
            years = np.arange(1, n_years + 1)

            # Determine breakthrough status for each year
            breakthrough_occurred = years >= breakthrough_time
            time_since_breakthrough = np.maximum(0, years - breakthrough_time)

            # Calculate recycling efficiency profile based on breakthrough
            recycling_efficiency = np.zeros(n_years)
            for i in range(n_years):
                if breakthrough_occurred[i]:
                    # After breakthrough: efficiency depends on time since breakthrough
                    # Early breakthrough: lower efficiency due to impurities
                    # Later: higher efficiency as system stabilizes
                    efficiency = self.breakthrough_physics.params.recycling_efficiency
                    time_factor = min(
                        1.0, time_since_breakthrough[i] / 5.0
                    )  # Stabilize over 5 years
                    recycling_efficiency[i] = efficiency * time_factor
                else:
                    # Before breakthrough: minimal recycling (only incidental CO₂)
                    recycling_efficiency[i] = 0.1  # 10% base efficiency

            # Estimate produced CO₂ based on injection and breakthrough timing
            # This is a simplified model - in practice would use reservoir simulation
            produced_tonne_estimate = np.zeros(n_years)
            for i in range(n_years):
                if breakthrough_occurred[i]:
                    # After breakthrough: significant CO₂ production
                    # Use GOR-based estimation
                    gor = self.breakthrough_physics.calculate_post_breakthrough_gor(
                        eor_params, time_since_breakthrough[i]
                    )
                    # A more realistic estimate of oil production is based on OOIP and recovery factor.
                    ooip_stb = reservoir_params.get(
                        "ooip_stb", 1e7
                    )  # Default to 10 million STB if not provided
                    recovery_factor = eor_params.get(
                        "recovery_factor", 0.1
                    )  # Default to 10% recovery
                    recoverable_oil_stb = ooip_stb * recovery_factor

                    # Assume constant production over the project life for simplicity.
                    annual_oil_production_stb = recoverable_oil_stb / n_years

                    # GOR is scf/stb. Produced gas in MSCF is STB * gor / 1000.
                    # Convert MSCF of CO2 to tonnes using co2_density_tonne_per_mscf
                    produced_co2_mscf = (annual_oil_production_stb * gor) / 1000.0
                    produced_tonne_estimate[i] = (
                        produced_co2_mscf * self.co2_density_tonne_per_mscf
                    )
                else:
                    # Before breakthrough: minimal CO₂ production
                    produced_tonne_estimate[i] = injected_tonne[i] * 0.05  # 5% dissolution

            # Calculate recyclable CO₂
            recycled_tonne = produced_tonne_estimate * recycling_efficiency

            return {
                "breakthrough_time_years": breakthrough_time,
                "recycling_efficiency_profile": recycling_efficiency,
                "produced_tonne_estimate": produced_tonne_estimate,
                "recycled_tonne_breakthrough_aware": recycled_tonne,
                "breakthrough_occurred": breakthrough_occurred,
            }

        except Exception as e:
            logger.warning(f"Breakthrough-aware recycling calculation failed: {e}")
            return None

    def generate_material_balance_graphs(
        self, material_balance_data: Dict[str, np.ndarray], title_suffix: str = ""
    ) -> Dict[str, go.Figure]:
        """
        Generate comprehensive material balance graphs.

        Args:
            material_balance_data: Output from calculate_material_balance
            title_suffix: Optional suffix for graph titles

        Returns:
            Dictionary of Plotly figures for different material balance views
        """
        years = material_balance_data["years"]

        # Main material balance chart
        fig_main = make_subplots(specs=[[{"secondary_y": True}]])

        # Primary Y-axis: Mass flows (tonnes)
        fig_main.add_trace(
            go.Bar(
                x=years,
                y=material_balance_data["injected_tonne"],
                name="CO₂ Injected",
                marker_color="blue",
            ),
            secondary_y=False,
        )
        fig_main.add_trace(
            go.Bar(
                x=years,
                y=material_balance_data["produced_tonne"],
                name="CO₂ Produced",
                marker_color="red",
            ),
            secondary_y=False,
        )
        fig_main.add_trace(
            go.Bar(
                x=years,
                y=material_balance_data["recycled_tonne"],
                name="CO₂ Recycled",
                marker_color="green",
            ),
            secondary_y=False,
        )
        fig_main.add_trace(
            go.Scatter(
                x=years,
                y=material_balance_data["net_stored_tonne"],
                name="Net Stored",
                line=dict(color="orange", width=3),
            ),
            secondary_y=False,
        )

        # Secondary Y-axis: Cumulative storage
        fig_main.add_trace(
            go.Scatter(
                x=years,
                y=material_balance_data["cumulative_stored_tonne"],
                name="Cumulative Stored",
                line=dict(color="purple", width=3, dash="dot"),
            ),
            secondary_y=True,
        )

        fig_main.update_layout(
            title=f"CO₂ Material Balance{title_suffix}",
            xaxis_title="Project Year",
            yaxis_title="CO₂ Mass (tonnes)",
            yaxis2_title="Cumulative Storage (tonnes)",
            barmode="group",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )

        # Storage efficiency chart
        fig_efficiency = go.Figure()
        fig_efficiency.add_trace(
            go.Scatter(
                x=years,
                y=material_balance_data["storage_efficiency"] * 100,
                name="Storage Efficiency",
                line=dict(color="blue", width=2),
            )
        )
        fig_efficiency.add_trace(
            go.Scatter(
                x=years,
                y=material_balance_data["annual_leakage_tonne"],
                name="Annual Leakage",
                line=dict(color="red", width=2),
                yaxis="y2",
            )
        )

        fig_efficiency.update_layout(
            title=f"Storage Efficiency and Leakage{title_suffix}",
            xaxis_title="Project Year",
            yaxis_title="Storage Efficiency (%)",
            yaxis2=dict(title="Leakage (tonnes)", overlaying="y", side="right"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )

        # Cumulative breakdown chart
        fig_cumulative = go.Figure()
        fig_cumulative.add_trace(
            go.Scatter(
                x=years,
                y=material_balance_data["cumulative_stored_tonne"],
                name="Cumulative Stored",
                line=dict(color="green", width=3),
            )
        )
        fig_cumulative.add_trace(
            go.Scatter(
                x=years,
                y=np.cumsum(material_balance_data["injected_tonne"]),
                name="Cumulative Injected",
                line=dict(color="blue", width=2, dash="dash"),
            )
        )
        fig_cumulative.add_trace(
            go.Scatter(
                x=years,
                y=np.cumsum(material_balance_data["produced_tonne"]),
                name="Cumulative Produced",
                line=dict(color="red", width=2, dash="dash"),
            )
        )

        fig_cumulative.update_layout(
            title=f"Cumulative CO₂ Balance{title_suffix}",
            xaxis_title="Project Year",
            yaxis_title="Cumulative CO₂ (tonnes)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )

        return {
            "main_balance": fig_main,
            "efficiency": fig_efficiency,
            "cumulative": fig_cumulative,
        }

    def generate_summary_statistics(
        self, material_balance_data: Dict[str, np.ndarray]
    ) -> Dict[str, float]:
        """
        Generate summary statistics for material balance analysis.

        Args:
            material_balance_data: Output from calculate_material_balance

        Returns:
            Dictionary of summary statistics
        """
        total_purchased_tonne = float(np.sum(material_balance_data["injected_tonne"]))
        total_recycled_tonne = float(np.sum(material_balance_data["recycled_tonne"]))
        total_gross_injected_tonne = total_purchased_tonne + total_recycled_tonne
        total_produced_tonne = float(np.sum(material_balance_data["produced_tonne"]))
        total_net_stored_tonne = float(np.sum(material_balance_data["net_stored_tonne"]))
        total_leakage_tonne = float(np.sum(material_balance_data["annual_leakage_tonne"]))
        uncaptured_tonne = max(0.0, total_produced_tonne - total_recycled_tonne)
        # Gross balance error: Gross Injected vs (Net Stored + Leakage + Produced)
        gross_accounted = total_net_stored_tonne + total_leakage_tonne + total_produced_tonne
        mb_error = abs(total_gross_injected_tonne - gross_accounted)

        return {
            "total_purchased_tonne": total_purchased_tonne,
            "total_injected_tonne": total_purchased_tonne,
            "total_gross_injected_tonne": total_gross_injected_tonne,
            "total_produced_tonne": total_produced_tonne,
            "total_recycled_tonne": total_recycled_tonne,
            "total_uncaptured_tonne": uncaptured_tonne,
            "total_net_stored_tonne": total_net_stored_tonne,
            "final_cumulative_stored_tonne": float(material_balance_data["cumulative_stored_tonne"][-1])
            if len(material_balance_data["cumulative_stored_tonne"]) > 0
            else 0.0,
            "avg_storage_efficiency": float(np.mean(material_balance_data["storage_efficiency"])),
            "total_leakage_tonne": total_leakage_tonne,
            "storage_efficiency_range": f"{np.min(material_balance_data['storage_efficiency'] * 100):.1f}%-{np.max(material_balance_data['storage_efficiency'] * 100):.1f}%",
            "mass_balance_error_tonne": mb_error,
        }


# Utility function to create material balance analysis from optimization results
def create_material_balance_from_optimization(
    optimization_results: Dict[str, any],
    resolution: str,
    co2_density_tonne_per_mscf: float = 0.053,
    leakage_rate_fraction: float = 0.01,
    reservoir_params: Optional[Dict[str, Any]] = None,
    eor_params: Optional[Dict[str, Any]] = None,
    eos_model: Optional[Any] = None,
) -> Dict[str, any]:
    """
    Convenience function to create material balance analysis from optimization results.

    Args:
        optimization_results: Results dictionary from OptimizationEngine
        resolution: Time resolution of the profiles (e.g., 'annual', 'monthly')
        co2_density_tonne_per_mscf: Density conversion factor
        leakage_rate_fraction: Annual leakage rate
        reservoir_params: Reservoir properties dict for breakthrough-aware recycling
        eor_params: EOR operation parameters dict for breakthrough-aware recycling
        eos_model: Optional ReservoirFluid EOS model for dynamic property calculation

    Returns:
        Complete material balance analysis with graphs and statistics
    """
    analyzer = MaterialBalanceAnalyzer(co2_density_tonne_per_mscf, eos_model=eos_model)

    # Extract CO2 profiles from optimization results
    profiles = optimization_results.get("optimized_profiles", {})

    injected_key = f"{resolution}_co2_purchased_mscf"
    if injected_key not in profiles:
        for fallback in ["yearly_co2_purchased_mscf", "annual_co2_purchased_mscf", "monthly_co2_purchased_mscf"]:
            if fallback in profiles:
                injected_key = fallback
                break

    produced_key = f"{resolution}_co2_produced_mscf"
    if produced_key not in profiles:
        for fallback in ["yearly_co2_produced_mscf", "annual_co2_produced_mscf", "monthly_co2_produced_mscf"]:
            if fallback in profiles:
                produced_key = fallback
                break

    recycled_key = f"{resolution}_co2_recycled_mscf"
    if recycled_key not in profiles:
        for fallback in ["yearly_co2_recycled_mscf", "annual_co2_recycled_mscf", "monthly_co2_recycled_mscf"]:
            if fallback in profiles:
                recycled_key = fallback
                break

    if not profiles or injected_key not in profiles:
        logger.warning(f"No CO2 injection profile found for key '{injected_key}' in optimization results")
        return {}

    # Calculate material balance with breakthrough-aware recycling
    balance_data = analyzer.calculate_material_balance(
        profiles[injected_key],
        profiles.get(produced_key, np.zeros_like(profiles[injected_key])),
        profiles.get(recycled_key, np.zeros_like(profiles[injected_key])),
        leakage_rate_fraction,
        reservoir_params=reservoir_params,
        eor_params=eor_params,
    )

    # If the profile was monthly, adjust the x-axis time vector to years
    if resolution == "monthly" or "monthly" in injected_key:
        balance_data["years"] = np.arange(1, len(balance_data["injected_tonne"]) + 1) / 12.0

    # Generate graphs
    graphs = analyzer.generate_material_balance_graphs(balance_data, " - Optimized Scenario")

    # Generate summary statistics
    stats = analyzer.generate_summary_statistics(balance_data)

    return {"material_balance_data": balance_data, "graphs": graphs, "summary_statistics": stats}
