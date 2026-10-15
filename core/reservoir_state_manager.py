"""
Centralized in-memory Shared Earth Model State Manager for CO2-EOR Optimizer.
Maintains dynamic coupling across Geology, PVT, Wells, Geomechanics, and Simulation.
"""

from copy import deepcopy
import logging
from typing import Any, Dict, List, Optional
import numpy as np

from PyQt6.QtCore import QObject, pyqtSignal

from core.data_models import (
    ReservoirData,
    PVTProperties,
    EORParameters,
    WellData,
    GeomechanicsParameters,
    EmpiricalFittingParameters,
    LayerDefinition,
    calculate_koval_from_reservoir,
)

logger = logging.getLogger(__name__)


class ReservoirStateManager(QObject):
    """
    Centralized in-memory Shared Earth Model state container.
    Dynamically synchronizes and propagates parameter changes across physical domains.
    """

    state_modified = pyqtSignal(str)  # Domain name that changed
    geology_updated = pyqtSignal(object)  # Emits ReservoirData
    pvt_updated = pyqtSignal(object)  # Emits PVTProperties
    wells_updated = pyqtSignal(list)  # Emits List[WellData]
    geomechanics_updated = pyqtSignal(object)  # Emits GeomechanicsParameters
    simulation_ready = pyqtSignal(bool)

    def __init__(self, parent: Optional[QObject] = None):
        super().__init__(parent)
        self._reservoir_data: Optional[ReservoirData] = None
        self._pvt_properties: Optional[PVTProperties] = None
        self._eor_parameters: Optional[EORParameters] = None
        self._well_data_list: List[WellData] = []
        self._geomechanics_parameters: Optional[GeomechanicsParameters] = None
        self._fitting_parameters: Optional[EmpiricalFittingParameters] = None
        self._is_verified: bool = False

    # -------------------------------------------------------------------------
    # Properties & Accessors
    # -------------------------------------------------------------------------
    @property
    def reservoir_data(self) -> Optional[ReservoirData]:
        return self._reservoir_data

    @property
    def pvt_properties(self) -> Optional[PVTProperties]:
        return self._pvt_properties

    @property
    def eor_parameters(self) -> Optional[EORParameters]:
        return self._eor_parameters

    @property
    def well_data_list(self) -> List[WellData]:
        return self._well_data_list

    @property
    def geomechanics_parameters(self) -> Optional[GeomechanicsParameters]:
        return self._geomechanics_parameters

    @property
    def fitting_parameters(self) -> Optional[EmpiricalFittingParameters]:
        return self._fitting_parameters

    @property
    def fault_properties(self) -> Optional[Dict[str, Any]]:
        if self._reservoir_data and getattr(self._reservoir_data, 'fault_properties', None):
            return self._reservoir_data.fault_properties
        return None

    @property
    def is_verified(self) -> bool:
        return self._is_verified

    @is_verified.setter
    def is_verified(self, val: bool) -> None:
        self._is_verified = bool(val)

    # -------------------------------------------------------------------------
    # Domain Setters & Dynamic Coupling
    # -------------------------------------------------------------------------
    def set_reservoir_data(self, data: Optional[ReservoirData], notify: bool = True) -> None:
        self._reservoir_data = data
        if self._reservoir_data and self._reservoir_data.layer_definitions:
            self._update_layer_heterogeneity()
        if notify:
            self.geology_updated.emit(self._reservoir_data)
            self.state_modified.emit("geology")
            self._check_simulation_readiness()

    def set_pvt_properties(self, data: Optional[PVTProperties], notify: bool = True) -> None:
        self._pvt_properties = data
        if notify:
            self.pvt_updated.emit(self._pvt_properties)
            self.state_modified.emit("pvt")
            self._check_simulation_readiness()

    def set_eor_parameters(self, data: Optional[EORParameters], notify: bool = True) -> None:
        self._eor_parameters = data
        if notify:
            self.state_modified.emit("eor")
            self._check_simulation_readiness()

    def set_well_data_list(self, wells: List[WellData], notify: bool = True) -> None:
        self._well_data_list = list(wells)
        if notify:
            self.wells_updated.emit(self._well_data_list)
            self.state_modified.emit("wells")
            self._check_simulation_readiness()

    def set_geomechanics_parameters(
        self, data: Optional[GeomechanicsParameters], notify: bool = True
    ) -> None:
        self._geomechanics_parameters = data
        if notify:
            self.geomechanics_updated.emit(self._geomechanics_parameters)
            self.state_modified.emit("geomechanics")
            self._check_simulation_readiness()

    def set_fitting_parameters(
        self, data: Optional[EmpiricalFittingParameters], notify: bool = True
    ) -> None:
        self._fitting_parameters = data
        if notify:
            self.state_modified.emit("fitting")

    # -------------------------------------------------------------------------
    # Layer Heterogeneity Coupling (Dykstra-Parsons & Koval Factor)
    # -------------------------------------------------------------------------
    def _update_layer_heterogeneity(self) -> None:
        """
        Calculates thickness-weighted average permeability and Dykstra-Parsons
        coefficient V_DP from layer definitions, updating ReservoirData.
        """
        if not self._reservoir_data or not self._reservoir_data.layer_definitions:
            return

        layers = self._reservoir_data.layer_definitions
        total_thickness = sum(layer.thickness for layer in layers if layer.thickness > 0)
        if total_thickness <= 0:
            return

        base_perm = self._reservoir_data.average_permeability or 100.0

        # Thickness-weighted average permeability
        weighted_k = sum(
            (layer.permeability_multiplier * base_perm) * layer.thickness
            for layer in layers
            if layer.thickness > 0
        )
        k_avg = weighted_k / total_thickness
        self._reservoir_data.average_permeability = float(k_avg)

        # Dykstra-Parsons heterogeneity coefficient
        k_values = [
            layer.permeability_multiplier * base_perm
            for layer in layers
            if layer.thickness > 0
        ]
        if len(k_values) >= 3:
            sorted_k = sorted(k_values, reverse=True)
            k_50 = float(np.percentile(sorted_k, 50))
            k_84_1 = float(np.percentile(sorted_k, 15.9))  # 1 standard deviation below median
            v_dp = (k_50 - k_84_1) / max(k_50, 1e-5)
            v_dp = float(np.clip(v_dp, 0.0, 0.95))
        elif len(k_values) == 2:
            k_max, k_min = max(k_values), min(k_values)
            v_dp = float(np.clip((k_max - k_min) / max(k_max, 1e-5), 0.0, 0.95))
        else:
            v_dp = 0.0

        if hasattr(self._reservoir_data, "v_dp_coefficient"):
            self._reservoir_data.v_dp_coefficient = v_dp

        logger.info(
            f"ReservoirStateManager: Updated layered model -> k_avg={k_avg:.1f} mD, V_DP={v_dp:.3f}"
        )

    # -------------------------------------------------------------------------
    # Simulation Readiness Validation
    # -------------------------------------------------------------------------
    def _check_simulation_readiness(self) -> bool:
        ready = (
            self._reservoir_data is not None
            and self._pvt_properties is not None
            and self._eor_parameters is not None
        )
        self.simulation_ready.emit(ready)
        return ready

    def get_full_project_snapshot(self) -> Dict[str, Any]:
        """Returns deep copies of all state components."""
        return {
            "reservoir_data": deepcopy(self._reservoir_data),
            "pvt_properties": deepcopy(self._pvt_properties),
            "eor_parameters": deepcopy(self._eor_parameters),
            "well_data_list": deepcopy(self._well_data_list),
            "geomechanics_parameters": deepcopy(self._geomechanics_parameters),
            "fitting_parameters": deepcopy(self._fitting_parameters),
            "is_verified": self._is_verified,
        }
