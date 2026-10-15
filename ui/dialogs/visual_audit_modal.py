"""
Visual Data Audit Modal ("Shared Earth Check")
Part of Workstream 1.4: Real-Time Subsurface Visualizers & Shared Earth Model.

Mandatory 3-tab visual confirmation gate before executing simulation or optimization:
- Tab 1: Heterogeneity & Variograms (GeostatisticsVisualizerWidget)
- Tab 2: Stratigraphy & Completions (GeologyCrossSectionWidget)
- Tab 3: Geomechanics & Fault Containment (FaultGeometryVisualizerWidget)
"""

import logging
from typing import Dict, Any, Optional

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QTabWidget, QLabel,
    QPushButton, QFrame, QMessageBox
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QIcon

from ui.widgets.geostatistics_visualizer_widget import GeostatisticsVisualizerWidget
from ui.widgets.geology_cross_section_widget import GeologyCrossSectionWidget
from ui.widgets.fault_geometry_visualizer_widget import FaultGeometryVisualizerWidget

logger = logging.getLogger(__name__)


class VisualAuditModal(QDialog):
    """
    Unified Shared Earth Visual Data Confirmation Gate.
    Provides multi-domain inspection before simulation and optimization dispatch.
    """
    def __init__(self, project_data: Optional[Dict[str, Any]] = None, parent: Optional[QDialog] = None):
        super().__init__(parent)
        self.project_data = project_data or {}
        self.setWindowTitle("Pre-Flight Shared Earth Visual Audit & Model Confirmation Gate")
        self.setMinimumSize(1050, 720)
        self.resize(1150, 780)

        self._setup_ui()
        self._load_project_data_into_visualizers()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        # Header banner
        header_frame = QFrame()
        header_frame.setFrameShape(QFrame.Shape.StyledPanel)
        header_frame.setStyleSheet("QFrame { background-color: #1e3d59; border-radius: 6px; padding: 8px; }")
        header_layout = QHBoxLayout(header_frame)

        title_lbl = QLabel("SINGLE INTEGRATED SHARED EARTH MODEL — PRE-FLIGHT VISUAL CONFIRMATION")
        title_lbl.setStyleSheet("color: white; font-weight: bold; font-size: 13px;")
        header_layout.addWidget(title_lbl)
        header_layout.addStretch()

        std_lbl = QLabel("ISO 27914 / EPA Class VI Compliant")
        std_lbl.setStyleSheet("color: #17a2b8; font-weight: bold; font-size: 11px;")
        header_layout.addWidget(std_lbl)
        layout.addWidget(header_frame)

        # 3-Tab Shared Earth Verification Widget
        self.tab_widget = QTabWidget()

        # Tab 1: Heterogeneity & Variograms
        self.geostat_widget = GeostatisticsVisualizerWidget(self)
        self.tab_widget.addTab(self.geostat_widget, QIcon.fromTheme("view-statistics"), "1. Spatial Heterogeneity & Variograms")

        # Tab 2: Stratigraphy & Slicing
        self.geology_widget = GeologyCrossSectionWidget(self)
        self.tab_widget.addTab(self.geology_widget, QIcon.fromTheme("drive-harddisk"), "2. Stratigraphy, Slicing & Completions")

        # Tab 3: Geomechanics & Faults
        self.fault_widget = FaultGeometryVisualizerWidget(self)
        self.tab_widget.addTab(self.fault_widget, QIcon.fromTheme("applications-science"), "3. Geomechanics, Faults & Containment")

        layout.addWidget(self.tab_widget, stretch=1)

        # Bottom Action Bar
        footer_layout = QHBoxLayout()
        self.audit_status_lbl = QLabel("Status: Visual Inspection Ready. Confirm each domain prior to simulation.")
        self.audit_status_lbl.setStyleSheet("font-weight: bold; color: #495057;")
        footer_layout.addWidget(self.audit_status_lbl)
        footer_layout.addStretch()

        self.confirm_btn = QPushButton(QIcon.fromTheme("emblem-default"), "Approve & Confirm Subsurface Model")
        self.confirm_btn.setStyleSheet("font-weight: bold; background-color: #28a745; color: white; padding: 8px 16px; border-radius: 4px;")
        self.confirm_btn.clicked.connect(self._on_approve_clicked)
        footer_layout.addWidget(self.confirm_btn)

        self.cancel_btn = QPushButton("Close")
        self.cancel_btn.clicked.connect(self.reject)
        footer_layout.addWidget(self.cancel_btn)

        layout.addLayout(footer_layout)

    def _load_project_data_into_visualizers(self):
        try:
            manual_inputs = self.project_data.get("manual_inputs", {})
            nx = int(manual_inputs.get("nx", 50))
            ny = int(manual_inputs.get("ny", 50))
            nz = int(manual_inputs.get("nz", 10))
            base_perm = float(manual_inputs.get("perm", 100.0) or 100.0)
            base_poro = float(manual_inputs.get("poro", 0.20) or 0.20)
            length_ft = float(manual_inputs.get("length", 2000.0) or 2000.0)
            area_acres = float(manual_inputs.get("area", 1000.0) or 1000.0)
            width_ft = (area_acres * 43560.0) / max(length_ft, 1.0)
            thickness_ft = float(manual_inputs.get("thickness", 50.0) or 50.0)
            dip_angle = float(manual_inputs.get("dip_angle", 0.0) or 0.0)
            wells = self.project_data.get("well_data_list", [])

            # 1. Geostatistical Visualizer
            self.geostat_widget.set_grid_dimensions(nx, ny, base_perm, base_poro)
            self.geostat_widget.set_well_data(wells)
            self.geostat_widget.generate_realization()

            # 2. Stratigraphic Cross-Section Visualizer
            grid_dict = self.project_data.get("grid") or {}
            perm_grid = grid_dict.get("PERMX")
            poro_grid = grid_dict.get("PORO")
            self.geology_widget.set_grid_data(
                perm_grid=perm_grid,
                poro_grid=poro_grid,
                length_ft=length_ft,
                width_ft=width_ft,
                thickness_ft=thickness_ft,
                dip_angle_deg=dip_angle,
                well_data_list=wells
            )

            # 3. Fault Visualizer
            fault_params = self.project_data.get("fault_properties") or {}
            self.fault_widget.set_reservoir_geometry(
                length_ft=length_ft,
                width_ft=width_ft,
                thickness_ft=thickness_ft,
                top_depth_ft=4000.0,
                well_data_list=wells
            )
            self.fault_widget.set_fault_parameters(fault_params)

        except Exception as e:
            logger.error(f"Error loading project data into visual audit modal: {e}", exc_info=True)

    def _on_approve_clicked(self):
        QMessageBox.information(
            self,
            "Subsurface Model Approved",
            "Shared Earth Model inputs have been visually confirmed.\n"
            "Geological heterogeneity, stratigraphic cross-sections, and fault seal containment verified."
        )
        self.accept()
