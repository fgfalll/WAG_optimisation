"""
Geology Engine for CO2 EOR Simulation (Deprecated).
Moved to deprecated as part of core cleanup.
"""

import warnings
import numpy as np
import logging

from core.data_models import PVTProperties, ReservoirData, EORParameters

logger = logging.getLogger(__name__)


class GeologyEngine:
    """
    Geology engine for CO2 EOR simulation (Deprecated).
    Handles geological calculations, heterogeneity, and sweep efficiency modifiers.
    """

    def __init__(self, reservoir: ReservoirData, pvt: PVTProperties, eor_params: EORParameters):
        warnings.warn(
            "GeologyEngine is deprecated and uncalibrated. Active displacement uses FastProfileGenerator and analytical models.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.reservoir = reservoir
        self.eor_params = eor_params

        # Calculate mobility ratio from PVT data
        oil_viscosity = pvt.oil_viscosity_cp or 2.0
        co2_viscosity = pvt.gas_viscosity_cp or 0.08
        self.mobility_ratio = oil_viscosity / (co2_viscosity + 1e-6)

    def calculate_geology_enhanced_sweep_efficiency(self) -> float:
        M = self.mobility_ratio
        if M <= 1.0:
            base_efficiency = 1.0
        else:
            if M <= 10:
                base_efficiency = 0.5 + 0.4 * np.log10(M) / (M - 1)
            else:
                base_efficiency = np.exp(-0.1 * (M - 10)) * (0.546 + 0.0357 / M)

        geology_factor = self._calculate_geology_factor()
        enhanced_efficiency = base_efficiency * geology_factor
        return float(np.clip(enhanced_efficiency, 0.1, 1.0))

    def _calculate_geology_factor(self) -> float:
        factor = 1.0
        rock_type = getattr(self.reservoir, "rock_type", None)
        if rock_type:
            if rock_type == "sandstone":
                factor *= 1.1
            elif rock_type == "carbonate":
                factor *= 0.9
            elif rock_type == "shale":
                factor *= 0.7

        depositional_env = getattr(self.reservoir, "depositional_environment", None)
        if depositional_env:
            if depositional_env in ["fluvial", "deltaic"]:
                factor *= 1.05
            elif depositional_env == "aeolian":
                factor *= 0.95
            elif depositional_env in ["deep_marine", "shallow_marine"]:
                factor *= 1.0

        structural_complexity = getattr(self.reservoir, "structural_complexity", None)
        if structural_complexity:
            if structural_complexity == "simple":
                factor *= 1.1
            elif structural_complexity == "moderate":
                factor *= 1.0
            elif structural_complexity == "complex":
                factor *= 0.9
            elif structural_complexity == "very_complex":
                factor *= 0.8

        if hasattr(self.reservoir, "geostatistical_grid") and self.reservoir.geostatistical_grid is not None:
            heterogeneity = self.calculate_heterogeneity_index()
            heterogeneity_penalty = 1.0 - (heterogeneity * 0.3)
            factor *= np.clip(heterogeneity_penalty, 0.7, 1.0)

        return float(factor)

    def calculate_heterogeneity_index(self) -> float:
        if self.reservoir.geostatistical_grid is None:
            return 0.0
        grid = self.reservoir.geostatistical_grid
        if grid.size == 0:
            return 0.0
        std_dev = np.std(grid)
        mean_val = np.mean(grid)
        return float(std_dev / mean_val) if mean_val > 0 else 0.0

    def get_geology_based_permeability_modifier(self) -> float:
        modifier = 1.0
        rock_type = getattr(self.reservoir, "rock_type", None)
        if rock_type:
            if rock_type == "sandstone":
                modifier *= 1.2
            elif rock_type == "carbonate":
                modifier *= 0.8
            elif rock_type == "shale":
                modifier *= 0.3
        return float(modifier)

    def calculate_geology_injection_factor(self) -> float:
        factor = 1.0
        rock_type = getattr(self.reservoir, "rock_type", None)
        if rock_type:
            if rock_type == "sandstone":
                factor *= 1.1
            elif rock_type == "carbonate":
                factor *= 0.9
            elif rock_type == "shale":
                factor *= 0.6

        structural_complexity = getattr(self.reservoir, "structural_complexity", None)
        if structural_complexity:
            if structural_complexity == "complex":
                factor *= 0.8
            elif structural_complexity == "very_complex":
                factor *= 0.6

        return float(np.clip(factor, 0.5, 1.5))
