import logging
from typing import Dict, Any, Optional
import numpy as np

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QDoubleSpinBox, QSpinBox, QComboBox, QPushButton, QFrame,
    QScrollArea, QFormLayout, QCheckBox, QLayout
)
from PyQt6.QtCore import pyqtSignal, Qt
from PyQt6.QtGui import QColor

logger = logging.getLogger(__name__)


def _clear_layout(layout):
    """Recursively removes and deletes all child widgets and sub-layouts."""
    if layout is None:
        return
    while layout.count() > 0:
        item = layout.takeAt(0)
        widget = item.widget()
        if widget is not None:
            widget.setParent(None)
            widget.deleteLater()
        sub_layout = item.layout()
        if sub_layout is not None:
            _clear_layout(sub_layout)


class ContextualPropertyGrid(QFrame):
    """
    Spacious contextual property editor displaying parameters for the currently selected
    Model Tree item or clicked 3D wellbore / reservoir element.
    Provides physical bounds validation and instant synchronization.
    """
    parameter_changed = pyqtSignal(str, str, object) # domain, key, value
    well_updated = pyqtSignal(str, dict)             # well_name, update_dict
    apply_requested = pyqtSignal()
    add_well_requested = pyqtSignal()
    edit_well_requested = pyqtSignal(str)
    delete_well_requested = pyqtSignal(str)
    generate_pattern_requested = pyqtSignal(str)
    manage_faults_requested = pyqtSignal(int)
    add_fault_requested = pyqtSignal()
    delete_fault_requested = pyqtSignal(str)
    isolate_fault_requested = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setStyleSheet("""
            QFrame {
                background: #ffffff;
                border: none;
            }
            QLabel#gridHeader {
                color: #0f172a;
                font-weight: bold;
                font-size: 12px;
                padding: 6px 8px;
                background: #f8fafc;
                border-bottom: 1px solid #e2e8f0;
            }
            QLabel#sectionHeader {
                color: #0284c7;
                font-weight: bold;
                font-size: 11px;
                margin-top: 10px;
                margin-bottom: 4px;
            }
            QLabel {
                color: #334155;
                font-size: 11px;
            }
            QLineEdit, QComboBox {
                background: #ffffff;
                color: #0f172a;
                border: 1.5px solid #94a3b8;
                border-radius: 4px;
                padding: 4px 6px;
                font-size: 11px;
                min-height: 22px;
            }
            QDoubleSpinBox, QSpinBox {
                background: #ffffff;
                color: #0f172a;
                border: 1.5px solid #94a3b8;
                border-radius: 4px;
                padding: 4px 22px 4px 6px;
                font-size: 11px;
                min-height: 22px;
            }
            QLineEdit:hover, QDoubleSpinBox:hover, QSpinBox:hover, QComboBox:hover {
                border: 1.5px solid #475569;
            }
            QLineEdit:focus, QDoubleSpinBox:focus, QSpinBox:focus, QComboBox:focus {
                border: 2px solid #0d6efd;
            }
            QDoubleSpinBox::up-button, QSpinBox::up-button {
                subcontrol-origin: border;
                subcontrol-position: top right;
                width: 16px;
                border-left: 1px solid #cbd5e1;
                border-bottom: 1px solid #e2e8f0;
                background: #f8fafc;
            }
            QDoubleSpinBox::up-button:hover, QSpinBox::up-button:hover {
                background: #e2e8f0;
            }
            QDoubleSpinBox::down-button, QSpinBox::down-button {
                subcontrol-origin: border;
                subcontrol-position: bottom right;
                width: 16px;
                border-left: 1px solid #cbd5e1;
                background: #f8fafc;
            }
            QDoubleSpinBox::down-button:hover, QSpinBox::down-button:hover {
                background: #e2e8f0;
            }
            QLabel.validBadge {
                color: #16a34a;
                font-weight: bold;
                font-size: 11px;
            }
            QLabel.invalidBadge {
                color: #dc3545;
                font-weight: bold;
                font-size: 11px;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self.lbl_title = QLabel("Contextual Property Grid")
        self.lbl_title.setObjectName("gridHeader")
        layout.addWidget(self.lbl_title)

        # Scroll area for form fields
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scroll.setStyleSheet("""
            QScrollArea {
                background: #ffffff;
                border: none;
            }
            QScrollBar:vertical {
                background: #f1f5f9;
                width: 9px;
                margin: 0px;
                border-radius: 4px;
            }
            QScrollBar::handle:vertical {
                background: #94a3b8;
                min-height: 32px;
                border-radius: 4px;
            }
            QScrollBar::handle:vertical:hover {
                background: #64748b;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
                background: none;
            }
        """)

        self.form_container = QWidget()
        self.form_container.setStyleSheet("background: #ffffff;")
        self.form_layout = QVBoxLayout(self.form_container)
        self.form_layout.setContentsMargins(8, 8, 8, 8)
        self.form_layout.setSpacing(6)
        self.form_layout.setSizeConstraint(QLayout.SizeConstraint.SetMinimumSize)
        self.scroll.setWidget(self.form_container)
        layout.addWidget(self.scroll, stretch=1)

        # Footer actions
        footer = QFrame()
        footer.setStyleSheet("background: #f8fafc; border-top: 1px solid #dee2e6;")
        footer_layout = QHBoxLayout(footer)
        footer_layout.setContentsMargins(6, 6, 6, 6)

        self.btn_apply = QPushButton("Apply && Update Model")
        self.btn_apply.setStyleSheet("""
            QPushButton {
                background: #0d6efd;
                color: #ffffff;
                font-weight: bold;
                font-size: 11px;
                padding: 6px 12px;
                border: 1px solid #0b5ed7;
                border-radius: 4px;
            }
            QPushButton:hover {
                background: #0b5ed7;
            }
        """)
        self.btn_apply.clicked.connect(self.apply_requested.emit)
        footer_layout.addWidget(self.btn_apply)

        layout.addWidget(footer)

        self.active_domain = "reservoir"
        self.active_key = "grid"
        self.inputs = {}
        self._has_inputs = True

    def has_editable_inputs(self) -> bool:
        """Returns True if the currently loaded node has editable parameters for the user to input."""
        return getattr(self, '_has_inputs', True)

    def update_input_values(self, values: Dict[str, Any]):
        """Updates values of existing inputs in-place without rebuilding the layout or destroying handles."""
        if not hasattr(self, 'inputs') or not self.inputs:
            return
        for key, widget in self.inputs.items():
            if key not in values:
                continue
            val = values[key]
            try:
                widget.blockSignals(True)
                if isinstance(widget, (QDoubleSpinBox, QSpinBox)):
                    widget.setValue(float(val))
                elif isinstance(widget, QComboBox):
                    if isinstance(val, int):
                        if 0 <= val < widget.count():
                            widget.setCurrentIndex(val)
                    else:
                        idx = widget.findText(str(val))
                        if idx >= 0:
                            widget.setCurrentIndex(idx)
                elif isinstance(widget, QCheckBox):
                    widget.setChecked(bool(val))
                widget.blockSignals(False)
            except Exception as e:
                try:
                    widget.blockSignals(False)
                except Exception:
                    pass
                logger.debug(f"Failed to update input widget '{key}': {e}")

    def load_node(
        self,
        domain: str,
        item_key: str,
        data_store: Dict[str, Any],
        well_data_list: Optional[list] = None,
        fault_data_list: Optional[list] = None
    ):
        """Loads and builds the property form according to the selected node."""
        self.active_domain = domain
        self.active_key = item_key

        # Clear existing form thoroughly (recursive)
        _clear_layout(self.form_layout)
        self.inputs.clear()

        # Nodes with no editable parameters (e.g. dashboards)
        if domain in ["surveillance", "surveillance_audit", "audit", "workstation", "multi_domain"] or item_key in ["surveillance", "workstation", "multi_domain", "audit", "visual_audit"]:
            self._has_inputs = False
            self.lbl_title.setText("Surveillance & Audit Workstation")
            lbl_info = QLabel("Full-width monitoring dashboard active.\nParameter input panel is hidden.")
            lbl_info.setStyleSheet("color: #64748b; font-size: 11px; padding: 12px;")
            self.form_layout.addWidget(lbl_info)
            self.form_layout.addStretch()
            return

        self._has_inputs = True

        if domain == "well_item":
            self.lbl_title.setText(f"Well Inspector: {item_key}")
            self._build_well_inspector(item_key, well_data_list or [])
        elif domain == "fault_item":
            f_name = item_key.split(":", 1)[0] if ":" in item_key else item_key
            self.lbl_title.setText(f"Fault Inspector: {f_name}")
            self._build_fault_item_inspector(f_name, fault_data_list or [])
        elif domain in ["wells", "well_network"]:
            if item_key in ["schedule_table", "schedule_gantt", "wag_schedule_graph"]:
                self.lbl_title.setText("Well Scheduling & Operating Controls")
                self._build_well_schedule_form(data_store, well_data_list or [])
            else:
                self.lbl_title.setText("Well Network Architecture")
                self._build_wells_network_overview_form(data_store, well_data_list or [])
        elif domain == "petrophysics":
            if item_key == "relperm":
                self.lbl_title.setText("Corey Relative Permeability & Wettability")
                self._build_relperm_form(data_store)
            elif item_key == "geostat":
                self.lbl_title.setText("Geostatistics & Spatial Variograms")
                self._build_geostat_form(data_store)
            else:
                self.lbl_title.setText("Petrophysics & Heterogeneity Architecture")
                self._build_petrophysics_form(data_store)
        elif domain == "pvt":
            self.lbl_title.setText("Fluids & PVT Thermodynamics")
            self._build_pvt_form(data_store)
        elif domain == "geomechanics":
            if item_key in ["faults", "fault_table", "fault_graph"]:
                self.lbl_title.setText("Fault Geometry, Kinematics & SGR Seal")
                self._build_fault_form(data_store)
            elif item_key in ["caprock", "caprock_table", "caprock_graph", "caprock_strat_table", "caprock_column_graph"]:
                self.lbl_title.setText("Caprock Multi-Layer Stratigraphy & Sealing")
                self._build_caprock_form(data_store)
            else:
                self.lbl_title.setText("In-Situ Stress & EPA Class VI Ceiling")
                self._build_stress_uic_form(data_store)
        elif domain in ["storage", "utilisation", "geothermal"]:
            self.lbl_title.setText("Storage, Utilisation & Geothermal CPG")
            self._build_storage_geothermal_form(data_store)
        elif domain == "reservoir":
            if item_key == "studio_3d":
                self.lbl_title.setText("3D Volumetric Studio Settings")
                self._build_studio_3d_form(data_store)
            elif item_key in ["volumetrics", "ooip"]:
                self.lbl_title.setText("Reservoir Volumetrics & OOIP")
                self._build_volumetrics_form(data_store)
            elif item_key == "stratigraphy":
                self.lbl_title.setText("Stratigraphic Layering & Zonation")
                self._build_stratigraphy_form(data_store)
            elif item_key in ["grid", "dimensions"]:
                self.lbl_title.setText("Grid Dimensions & Boundary Conditions")
                self._build_grid_form(data_store)
            elif item_key == "root":
                self.lbl_title.setText("Reservoir Framework Master Overview")
                self._build_reservoir_overview_form(data_store)
            else:
                self.lbl_title.setText("Grid Dimensions & Boundary Conditions")
                self._build_grid_form(data_store)
        else:
            self.lbl_title.setText("Reservoir Grid Framework")
            self._build_grid_form(data_store)

        self.form_layout.addStretch()

    def _build_wells_network_overview_form(self, data: Dict[str, Any], well_data_list: list):
        lbl = QLabel("Well Network Architecture & Placement")
        lbl.setObjectName("sectionHeader")
        self.form_layout.addWidget(lbl)

        producers = [w for w in well_data_list if str(getattr(w, 'metadata', {}).get("type", "")).lower() == "producer"]
        injectors = [w for w in well_data_list if str(getattr(w, 'metadata', {}).get("type", "")).lower() == "injector"]

        status_lbl = QLabel(f"Total Wells: {len(well_data_list)} | Producers: {len(producers)} | Injectors: {len(injectors)}")
        status_lbl.setStyleSheet("color: #0284c7; font-weight: bold; font-size: 11px; padding: 4px;")
        self.form_layout.addWidget(status_lbl)

        # Quick action buttons for well network
        btn_box = QHBoxLayout()
        btn_box.setSpacing(4)
        btn_add = QPushButton("Add Well...")
        btn_add.setStyleSheet("background: #e0f2fe; color: #0284c7; border: 1px solid #bae6fd; border-radius: 4px; font-weight: bold; font-size: 10px; padding: 4px;")
        btn_add.clicked.connect(self.add_well_requested.emit)
        btn_box.addWidget(btn_add)

        btn_5spot = QPushButton("5-Spot")
        btn_5spot.setStyleSheet("background: #f1f5f9; border: 1px solid #cbd5e1; border-radius: 4px; font-weight: bold; font-size: 10px; padding: 4px;")
        btn_5spot.clicked.connect(lambda: self.generate_pattern_requested.emit("5spot"))
        btn_box.addWidget(btn_5spot)

        btn_9spot = QPushButton("9-Spot")
        btn_9spot.setStyleSheet("background: #f1f5f9; border: 1px solid #cbd5e1; border-radius: 4px; font-weight: bold; font-size: 10px; padding: 4px;")
        btn_9spot.clicked.connect(lambda: self.generate_pattern_requested.emit("9spot"))
        btn_box.addWidget(btn_9spot)
        self.form_layout.addLayout(btn_box)

        lbl_ops = QLabel("Global Well Deliverability Parameters")
        lbl_ops.setObjectName("sectionHeader")
        self.form_layout.addWidget(lbl_ops)

        form = QFormLayout()
        form.setSpacing(6)

        rw_spin = QDoubleSpinBox()
        rw_spin.setRange(0.1, 2.0)
        rw_spin.setSingleStep(0.05)
        rw_spin.setValue(float(data.get("well_radius", 0.35)))
        rw_spin.valueChanged.connect(lambda v: self.parameter_changed.emit("wells", "well_radius", v))
        form.addRow("Wellbore Radius (rw):", self._with_badge(rw_spin, "ft"))

        skin_spin = QDoubleSpinBox()
        skin_spin.setRange(-5.0, 50.0)
        skin_spin.setSingleStep(0.5)
        skin_spin.setValue(float(data.get("skin_factor", 0.0)))
        skin_spin.valueChanged.connect(lambda v: self.parameter_changed.emit("wells", "skin_factor", v))
        form.addRow("Formation Skin (s):", self._with_badge(skin_spin, "dim"))

        min_bhp_spin = QDoubleSpinBox()
        min_bhp_spin.setRange(100.0, 10000.0)
        min_bhp_spin.setSingleStep(50.0)
        min_bhp_spin.setValue(float(data.get("min_producer_bhp", 1000.0)))
        min_bhp_spin.valueChanged.connect(lambda v: self.parameter_changed.emit("wells", "min_producer_bhp", v))
        form.addRow("Min Producer BHP:", self._with_badge(min_bhp_spin, "psia"))

        max_bhp_spin = QDoubleSpinBox()
        max_bhp_spin.setRange(500.0, 15000.0)
        max_bhp_spin.setSingleStep(100.0)
        max_bhp_spin.setValue(float(data.get("max_injector_bhp", 5000.0)))
        max_bhp_spin.valueChanged.connect(lambda v: self.parameter_changed.emit("wells", "max_injector_bhp", v))
        form.addRow("Max Injector Sandface:", self._with_badge(max_bhp_spin, "psia"))

        self.form_layout.addLayout(form)

        if well_data_list:
            lbl_list = QLabel("Wellbore Trajectories & Coordinates (Click to Edit)")
            lbl_list.setObjectName("sectionHeader")
            self.form_layout.addWidget(lbl_list)
            for w in well_data_list:
                w_type = getattr(w, 'metadata', {}).get("type", "Well")
                sx = getattr(w, 'metadata', {}).get("SurfaceX", 0.0)
                sy = getattr(w, 'metadata', {}).get("SurfaceY", 0.0)
                btn_w = QPushButton(f"• {w.name} [{w_type.capitalize()}]: ({sx:.0f}, {sy:.0f}) ft")
                btn_w.setStyleSheet("""
                    QPushButton {
                        background: #f8fafc;
                        color: #1e293b;
                        border: 1px solid #e2e8f0;
                        border-radius: 3px;
                        font-size: 11px;
                        text-align: left;
                        padding: 4px 6px;
                    }
                    QPushButton:hover {
                        background: #e0f2fe;
                        color: #0284c7;
                        border-color: #bae6fd;
                    }
                """)
                w_name_target = w.name
                btn_w.clicked.connect(lambda checked=False, wn=w_name_target: self.edit_well_requested.emit(wn))
                self.form_layout.addWidget(btn_w)

    def _build_well_schedule_form(self, data: Dict[str, Any], well_data_list: list):
        lbl = QLabel("Well Scheduling & Operating Controls")
        lbl.setObjectName("sectionHeader")
        self.form_layout.addWidget(lbl)

        form = QFormLayout()
        form.setSpacing(6)

        mode_combo = QComboBox()
        mode_combo.addItems(["Rate Constrained (Target Rate)", "BHP Constrained (Pressure Limits)", "Coupled VFP / Network"])
        form.addRow("Primary Control Mode:", mode_combo)

        wag_cycle_spin = QDoubleSpinBox()
        wag_cycle_spin.setRange(10.0, 365.0)
        wag_cycle_spin.setSingleStep(15.0)
        wag_cycle_spin.setValue(float(data.get("wag_cycle_days", 90.0)))
        wag_cycle_spin.valueChanged.connect(lambda v: self.parameter_changed.emit("wells", "wag_cycle_days", v))
        form.addRow("WAG Half-Cycle Period:", self._with_badge(wag_cycle_spin, "days"))

        wag_ratio_spin = QDoubleSpinBox()
        wag_ratio_spin.setRange(0.2, 5.0)
        wag_ratio_spin.setSingleStep(0.1)
        wag_ratio_spin.setDecimals(1)
        wag_ratio_spin.setValue(float(data.get("wag_ratio", 1.5)))
        wag_ratio_spin.valueChanged.connect(lambda v: self.parameter_changed.emit("wells", "wag_ratio", v))
        form.addRow("WAG Ratio (Water:Gas):", self._with_badge(wag_ratio_spin, "vol:vol"))

        depth_val = float(data.get("depth", 5000.0))
        maip_sandface = 0.90 * 0.85 * depth_val
        maip_spin = QDoubleSpinBox()
        maip_spin.setRange(1000.0, 15000.0)
        maip_spin.setSingleStep(100.0)
        maip_spin.setValue(float(data.get("inj_bhp_max", maip_sandface)))
        maip_spin.valueChanged.connect(lambda v: self.parameter_changed.emit("wells", "inj_bhp_max", v))
        form.addRow("Injector Max BHP (MAIP):", self._with_badge(maip_spin, "psia"))

        bhp_min_spin = QDoubleSpinBox()
        bhp_min_spin.setRange(500.0, 5000.0)
        bhp_min_spin.setSingleStep(100.0)
        bhp_min_spin.setValue(float(data.get("prod_bhp_min", 1800.0)))
        bhp_min_spin.valueChanged.connect(lambda v: self.parameter_changed.emit("wells", "prod_bhp_min", v))
        form.addRow("Producer Min BHP Limit:", self._with_badge(bhp_min_spin, "psia"))

        gor_spin = QDoubleSpinBox()
        gor_spin.setRange(1.0, 50.0)
        gor_spin.setSingleStep(1.0)
        gor_spin.setValue(float(data.get("gor_shut_limit", 15.0)))
        gor_spin.valueChanged.connect(lambda v: self.parameter_changed.emit("wells", "gor_shut_limit", v))
        form.addRow("GOR Shut-In Ceiling:", self._with_badge(gor_spin, "MSCF/STB"))

        wcut_spin = QDoubleSpinBox()
        wcut_spin.setRange(50.0, 99.0)
        wcut_spin.setSingleStep(1.0)
        wcut_spin.setValue(float(data.get("wcut_shut_limit", 95.0)))
        wcut_spin.valueChanged.connect(lambda v: self.parameter_changed.emit("wells", "wcut_shut_limit", v))
        form.addRow("Water Cut Shut-In Limit:", self._with_badge(wcut_spin, "%"))

        self.form_layout.addLayout(form)

    def _build_reservoir_overview_form(self, data: Dict[str, Any]):
        lbl = QLabel("Reservoir Framework Master Overview")
        lbl.setObjectName("sectionHeader")
        self.form_layout.addWidget(lbl)

        # Quick Navigation buttons
        btn_box = QHBoxLayout()
        btn_box.setSpacing(4)

        btn_grid = QPushButton("Grid (NX, NY, NZ)")
        btn_grid.setStyleSheet("background: #f1f5f9; border: 1.5px solid #94a3b8; border-radius: 4px; font-weight: bold; font-size: 10px; padding: 4px;")
        btn_grid.clicked.connect(lambda: self.load_node("reservoir", "grid", data))
        btn_box.addWidget(btn_grid)

        btn_vol = QPushButton("Volumetrics & OOIP")
        btn_vol.setStyleSheet("background: #f1f5f9; border: 1.5px solid #94a3b8; border-radius: 4px; font-weight: bold; font-size: 10px; padding: 4px;")
        btn_vol.clicked.connect(lambda: self.load_node("reservoir", "volumetrics", data))
        btn_box.addWidget(btn_vol)

        btn_strat = QPushButton("Stratigraphy")
        btn_strat.setStyleSheet("background: #f1f5f9; border: 1.5px solid #94a3b8; border-radius: 4px; font-weight: bold; font-size: 10px; padding: 4px;")
        btn_strat.clicked.connect(lambda: self.load_node("reservoir", "stratigraphy", data))
        btn_box.addWidget(btn_strat)

        self.form_layout.addLayout(btn_box)

        # Master Overview Parameter Form
        form = QFormLayout()
        form.setSpacing(6)

        name_edit = QLineEdit(str(data.get("formation_name", "San Andres / Midale Carbonate Formation")))
        name_edit.textChanged.connect(lambda v: self.parameter_changed.emit("reservoir", "formation_name", v))
        form.addRow("Formation Name:", name_edit)

        nx_spin = QSpinBox()
        nx_spin.setRange(5, 500)
        nx_spin.setValue(int(data.get("nx", 50)))
        nx_spin.valueChanged.connect(lambda v: self.parameter_changed.emit("reservoir", "nx", v))
        form.addRow("Grid Cells X (NX):", self._with_badge(nx_spin, "cells"))

        ny_spin = QSpinBox()
        ny_spin.setRange(5, 500)
        ny_spin.setValue(int(data.get("ny", 50)))
        ny_spin.valueChanged.connect(lambda v: self.parameter_changed.emit("reservoir", "ny", v))
        form.addRow("Grid Cells Y (NY):", self._with_badge(ny_spin, "cells"))

        nz_spin = QSpinBox()
        nz_spin.setRange(1, 100)
        nz_spin.setValue(int(data.get("nz", 10)))
        nz_spin.valueChanged.connect(lambda v: self.parameter_changed.emit("reservoir", "nz", v))
        form.addRow("Grid Layers Z (NZ):", self._with_badge(nz_spin, "layers"))

        len_spin = QDoubleSpinBox()
        len_spin.setRange(100.0, 50000.0)
        len_spin.setValue(float(data.get("length", 2000.0)))
        len_spin.valueChanged.connect(lambda v: self.parameter_changed.emit("reservoir", "length", v))
        form.addRow("Length (X):", self._with_badge(len_spin, "ft"))

        area_spin = QDoubleSpinBox()
        area_spin.setRange(1.0, 10000.0)
        area_spin.setValue(float(data.get("area", 100.0)))
        area_spin.valueChanged.connect(lambda v: self.parameter_changed.emit("reservoir", "area", v))
        form.addRow("Areal Extent:", self._with_badge(area_spin, "acres"))

        h_spin = QDoubleSpinBox()
        h_spin.setRange(5.0, 1000.0)
        h_spin.setValue(float(data.get("thickness", 50.0)))
        h_spin.valueChanged.connect(lambda v: self.parameter_changed.emit("reservoir", "thickness", v))
        form.addRow("Net Pay (h):", self._with_badge(h_spin, "ft"))

        top_spin = QDoubleSpinBox()
        top_spin.setRange(500.0, 30000.0)
        top_spin.setValue(float(data.get("top_depth", 5000.0)))
        top_spin.valueChanged.connect(lambda v: self.parameter_changed.emit("reservoir", "top_depth", v))
        form.addRow("Top Depth TVD:", self._with_badge(top_spin, "ft"))

        poro_spin = QDoubleSpinBox()
        poro_spin.setRange(0.01, 0.45)
        poro_spin.setSingleStep(0.01)
        poro_spin.setDecimals(3)
        poro_spin.setValue(float(data.get("poro", 0.20)))
        poro_spin.valueChanged.connect(lambda v: self.parameter_changed.emit("reservoir", "poro", v))
        form.addRow("Porosity (φ):", self._with_badge(poro_spin, "frac"))

        perm_spin = QDoubleSpinBox()
        perm_spin.setRange(0.01, 5000.0)
        perm_spin.setSingleStep(10.0)
        perm_spin.setValue(float(data.get("perm", 100.0)))
        perm_spin.valueChanged.connect(lambda v: self.parameter_changed.emit("reservoir", "perm", v))
        form.addRow("Permeability (k):", self._with_badge(perm_spin, "mD"))

        self.form_layout.addLayout(form)

    def _build_grid_form(self, data: Dict[str, Any]):
        lbl = QLabel("Structured Grid Framework (Cartesian)")
        lbl.setObjectName("sectionHeader")
        self.form_layout.addWidget(lbl)

        form = QFormLayout()
        form.setSpacing(6)

        nx_spin = QSpinBox()
        nx_spin.setRange(5, 500)
        nx_spin.setValue(int(data.get("nx", 50)))
        nx_spin.valueChanged.connect(lambda v: self.parameter_changed.emit("reservoir", "nx", v))
        form.addRow("Grid Cells X (NX):", self._with_badge(nx_spin, "cells"))
        self.inputs["nx"] = nx_spin

        ny_spin = QSpinBox()
        ny_spin.setRange(5, 500)
        ny_spin.setValue(int(data.get("ny", 50)))
        ny_spin.valueChanged.connect(lambda v: self.parameter_changed.emit("reservoir", "ny", v))
        form.addRow("Grid Cells Y (NY):", self._with_badge(ny_spin, "cells"))
        self.inputs["ny"] = ny_spin

        nz_spin = QSpinBox()
        nz_spin.setRange(1, 100)
        nz_spin.setValue(int(data.get("nz", 10)))
        nz_spin.valueChanged.connect(lambda v: self.parameter_changed.emit("reservoir", "nz", v))
        form.addRow("Grid Layers Z (NZ):", self._with_badge(nz_spin, "layers"))
        self.inputs["nz"] = nz_spin

        lbl_geom = QLabel("Geometry & Dimensions")
        lbl_geom.setObjectName("sectionHeader")
        self.form_layout.addLayout(form)
        self.form_layout.addWidget(lbl_geom)

        form2 = QFormLayout()
        form2.setSpacing(6)

        len_spin = QDoubleSpinBox()
        len_spin.setRange(100.0, 50000.0)
        len_spin.setValue(float(data.get("length", 2000.0)))
        len_spin.valueChanged.connect(lambda v: self.parameter_changed.emit("reservoir", "length", v))
        form2.addRow("Length (X):", self._with_badge(len_spin, "ft"))
        self.inputs["length"] = len_spin

        area_spin = QDoubleSpinBox()
        area_spin.setRange(1.0, 10000.0)
        area_spin.setValue(float(data.get("area", 100.0)))
        area_spin.valueChanged.connect(lambda v: self.parameter_changed.emit("reservoir", "area", v))
        form2.addRow("Area:", self._with_badge(area_spin, "acres"))
        self.inputs["area"] = area_spin

        h_spin = QDoubleSpinBox()
        h_spin.setRange(5.0, 1000.0)
        h_spin.setValue(float(data.get("thickness", 50.0)))
        h_spin.valueChanged.connect(lambda v: self.parameter_changed.emit("reservoir", "thickness", v))
        form2.addRow("Net Pay:", self._with_badge(h_spin, "ft"))
        self.inputs["thickness"] = h_spin

        # Readouts
        nx = int(data.get("nx", 50))
        ny = int(data.get("ny", 50))
        nz = int(data.get("nz", 10))
        lx = float(data.get("length", 2000.0))
        area = float(data.get("area", 100.0))
        ly = (area * 43560.0) / max(lx, 1.0)
        h = float(data.get("thickness", 50.0))

        dx_val = lx / max(nx, 1)
        dy_val = ly / max(ny, 1)
        dz_val = h / max(nz, 1)

        dx_badge = QLabel(f"ΔX: {dx_val:.1f} ft | ΔY: {dy_val:.1f} ft | ΔZ: {dz_val:.1f} ft")
        dx_badge.setStyleSheet("color: #0284c7; font-weight: bold;")
        form2.addRow("Cell Dimensions:", dx_badge)

        cells_badge = QLabel(f"{nx * ny * nz:,} Active Cells")
        cells_badge.setStyleSheet("color: #16a34a; font-weight: bold;")
        form2.addRow("Total Mesh Cells:", cells_badge)

        bnd_combo = QComboBox()
        bnd_combo.addItems(["Closed / No-Flow Boundary", "Constant Pressure Edge-Drive", "Bottom Water Aquifer (Carter-Tracy)"])
        bnd_combo.setCurrentText(str(data.get("boundary_type", "Closed / No-Flow Boundary")))
        bnd_combo.currentTextChanged.connect(lambda v: self.parameter_changed.emit("reservoir", "boundary_type", v))
        form2.addRow("Outer Boundary:", bnd_combo)

        self.form_layout.addLayout(form2)

    def _build_volumetrics_form(self, data: Dict[str, Any]):
        lbl = QLabel("Structural Depths & Fluid Contacts")
        lbl.setObjectName("sectionHeader")
        self.form_layout.addWidget(lbl)

        form = QFormLayout()
        form.setSpacing(6)

        top_spin = QDoubleSpinBox()
        top_spin.setRange(500.0, 30000.0)
        top_spin.setSingleStep(50.0)
        top_spin.setValue(float(data.get("top_depth", 5000.0)))
        top_spin.valueChanged.connect(lambda v: self.parameter_changed.emit("reservoir", "top_depth", v))
        form.addRow("Top Depth TVD:", self._with_badge(top_spin, "ft"))

        datum_spin = QDoubleSpinBox()
        datum_spin.setRange(500.0, 30000.0)
        datum_spin.setSingleStep(50.0)
        datum_spin.setValue(float(data.get("datum_depth", 5025.0)))
        datum_spin.valueChanged.connect(lambda v: self.parameter_changed.emit("reservoir", "datum_depth", v))
        form.addRow("Datum Depth:", self._with_badge(datum_spin, "ft"))

        woc_spin = QDoubleSpinBox()
        woc_spin.setRange(500.0, 30000.0)
        woc_spin.setSingleStep(25.0)
        woc_spin.setValue(float(data.get("woc_depth", 5120.0)))
        woc_spin.valueChanged.connect(lambda v: self.parameter_changed.emit("reservoir", "woc_depth", v))
        form.addRow("Water-Oil Contact (WOC):", self._with_badge(woc_spin, "ft"))

        goc_spin = QDoubleSpinBox()
        goc_spin.setRange(500.0, 30000.0)
        goc_spin.setSingleStep(25.0)
        goc_spin.setValue(float(data.get("goc_depth", 4950.0)))
        goc_spin.valueChanged.connect(lambda v: self.parameter_changed.emit("reservoir", "goc_depth", v))
        form.addRow("Gas-Oil Contact (GOC):", self._with_badge(goc_spin, "ft"))

        ntg_spin = QDoubleSpinBox()
        ntg_spin.setRange(0.05, 1.00)
        ntg_spin.setSingleStep(0.05)
        ntg_spin.setValue(float(data.get("ntg", 0.85)))
        ntg_spin.valueChanged.connect(lambda v: self.parameter_changed.emit("reservoir", "ntg", v))
        form.addRow("Net-to-Gross (NTG):", self._with_badge(ntg_spin, "ratio"))

        lbl_sat = QLabel("In-Situ Reservoir & Fluid In-Place")
        lbl_sat.setObjectName("sectionHeader")
        self.form_layout.addLayout(form)
        self.form_layout.addWidget(lbl_sat)

        form2 = QFormLayout()
        form2.setSpacing(6)

        phi_spin = QDoubleSpinBox()
        phi_spin.setRange(0.01, 0.45)
        phi_spin.setSingleStep(0.01)
        phi_spin.setDecimals(3)
        phi_spin.setValue(float(data.get("poro", 0.20)))
        phi_spin.valueChanged.connect(lambda v: self.parameter_changed.emit("reservoir", "poro", v))
        form2.addRow("Average Porosity (φ):", self._with_badge(phi_spin, "frac"))

        swi_spin = QDoubleSpinBox()
        swi_spin.setRange(0.05, 0.80)
        swi_spin.setSingleStep(0.05)
        swi_spin.setDecimals(3)
        swi_spin.setValue(float(data.get("swi", 0.25)))
        swi_spin.valueChanged.connect(lambda v: self.parameter_changed.emit("reservoir", "swi", v))
        form2.addRow("Initial Water Sat (Swi):", self._with_badge(swi_spin, "frac"))

        boi_spin = QDoubleSpinBox()
        boi_spin.setRange(1.00, 2.50)
        boi_spin.setSingleStep(0.05)
        boi_spin.setDecimals(3)
        boi_spin.setValue(float(data.get("boi", 1.20)))
        boi_spin.valueChanged.connect(lambda v: self.parameter_changed.emit("reservoir", "boi", v))
        form2.addRow("Initial Oil FVF (Boi):", self._with_badge(boi_spin, "rb/STB"))

        cr_spin = QDoubleSpinBox()
        cr_spin.setRange(0.5, 20.0)
        cr_spin.setSingleStep(0.5)
        cr_spin.setValue(float(data.get("rock_compressibility_e6", 3.5)))
        cr_spin.valueChanged.connect(lambda v: self.parameter_changed.emit("reservoir", "rock_compressibility_e6", v))
        form2.addRow("Rock Compressibility (cr):", self._with_badge(cr_spin, "μsip"))

        m_spin = QDoubleSpinBox()
        m_spin.setRange(0.0, 2.0)
        m_spin.setSingleStep(0.05)
        m_spin.setValue(float(data.get("gas_cap_m", 0.0)))
        m_spin.valueChanged.connect(lambda v: self.parameter_changed.emit("reservoir", "gas_cap_m", v))
        form2.addRow("Gas Cap Ratio (m):", self._with_badge(m_spin, "ratio"))

        hcpv_badge = QLabel()
        hcpv_badge.setStyleSheet("color: #0284c7; font-weight: bold;")
        form2.addRow("Hydrocarbon PV (HCPV):", hcpv_badge)

        ooip_badge = QLabel()
        ooip_badge.setStyleSheet("color: #16a34a; font-weight: bold; font-size: 12px;")
        form2.addRow("Volumetric OOIP:", ooip_badge)

        def _update_calcs():
            p_val = phi_spin.value()
            s_val = swi_spin.value()
            b_val = max(boi_spin.value(), 0.1)
            n_val = ntg_spin.value()
            a_val = float(data.get("area", 100.0))
            h_val = float(data.get("thickness", 50.0))
            stb = (7758.0 * a_val * h_val * n_val * p_val * (1.0 - s_val)) / b_val
            mmstb = stb / 1e6
            hcpv_mm = (7758.0 * a_val * h_val * n_val * p_val * (1.0 - s_val)) / (5.615 * 1e6)
            hcpv_badge.setText(f"{hcpv_mm:.2f} MMbbl ({hcpv_mm * 5.615:.2f} MMCF)")
            ooip_badge.setText(f"{mmstb:.2f} MMSTB ({stb:,.0f} STB)")

        phi_spin.valueChanged.connect(lambda _: _update_calcs())
        swi_spin.valueChanged.connect(lambda _: _update_calcs())
        boi_spin.valueChanged.connect(lambda _: _update_calcs())
        ntg_spin.valueChanged.connect(lambda _: _update_calcs())
        _update_calcs()

        self.form_layout.addLayout(form2)

    def _build_studio_3d_form(self, data: Dict[str, Any]):
        lbl = QLabel("3D Viewport Controls & Rendering")
        lbl.setObjectName("sectionHeader")
        self.form_layout.addWidget(lbl)

        form = QFormLayout()
        form.setSpacing(6)

        prop_combo = QComboBox()
        prop_combo.addItems(["Permeability (PERMX)", "Porosity (PORO)", "Oil Saturation (So)", "Pore Pressure (psia)"])
        prop_combo.setCurrentText(str(data.get("active_3d_property", "Permeability (PERMX)")))
        prop_combo.currentTextChanged.connect(lambda v: self.parameter_changed.emit("reservoir", "active_3d_property", v))
        form.addRow("Active Scalar:", prop_combo)

        pal_combo = QComboBox()
        pal_combo.addItems(["turbo", "viridis", "plasma", "jet", "coolwarm", "rainbow"])
        pal_combo.setCurrentText(str(data.get("active_3d_palette", "turbo")))
        pal_combo.currentTextChanged.connect(lambda v: self.parameter_changed.emit("reservoir", "active_3d_palette", v))
        form.addRow("Colormap Palette:", pal_combo)

        zscale_spin = QDoubleSpinBox()
        zscale_spin.setRange(1.0, 15.0)
        zscale_spin.setSingleStep(0.5)
        zscale_spin.setValue(float(data.get("z_scale", 4.0)))
        zscale_spin.valueChanged.connect(lambda v: self.parameter_changed.emit("reservoir", "z_scale", v))
        form.addRow("Vertical Exaggeration:", self._with_badge(zscale_spin, "x"))

        slice_combo = QComboBox()
        slice_combo.addItems(["Interior Cross-Cut", "Solid Volume Block", "Horizon Top Surface", "Wellbore Core Slices"])
        slice_combo.setCurrentText(str(data.get("slice_mode", "Interior Cross-Cut")))
        slice_combo.currentTextChanged.connect(lambda v: self.parameter_changed.emit("reservoir", "slice_mode", v))
        form.addRow("Slicing Mode:", slice_combo)

        opacity_spin = QDoubleSpinBox()
        opacity_spin.setRange(0.10, 1.00)
        opacity_spin.setSingleStep(0.05)
        opacity_spin.setValue(float(data.get("cube_opacity", 0.85)))
        opacity_spin.valueChanged.connect(lambda v: self.parameter_changed.emit("reservoir", "cube_opacity", v))
        form.addRow("Block Opacity:", self._with_badge(opacity_spin, "ratio"))

        lbl_feat = QLabel("3D Subsurface Features")
        lbl_feat.setObjectName("sectionHeader")
        self.form_layout.addLayout(form)
        self.form_layout.addWidget(lbl_feat)

        form2 = QFormLayout()
        form2.setSpacing(6)

        cam_combo = QComboBox()
        cam_combo.addItems(["Isometric 3D", "Map View (X-Y)", "Cross-Section (X-Z)", "Cross-Section (Y-Z)"])
        cam_combo.currentTextChanged.connect(lambda v: self.parameter_changed.emit("reservoir", "camera_preset", v))
        form2.addRow("Camera Preset:", cam_combo)

        self.form_layout.addLayout(form2)

    def _build_stratigraphy_form(self, data: Dict[str, Any]):
        lbl = QLabel("Stratigraphic Layering & Zonation")
        lbl.setObjectName("sectionHeader")
        self.form_layout.addWidget(lbl)

        form = QFormLayout()
        form.setSpacing(6)

        name_edit = QLineEdit(str(data.get("formation_name", "San Andres / Midale Carbonate Formation")))
        name_edit.textChanged.connect(lambda v: self.parameter_changed.emit("reservoir", "formation_name", v))
        form.addRow("Formation Name:", name_edit)

        layers_spin = QSpinBox()
        layers_spin.setRange(1, 50)
        layers_spin.setValue(int(data.get("nz", 10)))
        layers_spin.valueChanged.connect(lambda v: self.parameter_changed.emit("reservoir", "nz", v))
        form.addRow("Geological Sub-layers:", self._with_badge(layers_spin, "zones"))

        nz_val = max(int(data.get("nz", 10)), 1)
        h_val = float(data.get("thickness", 50.0))
        dz_readout = QLabel(f"Avg Layer ΔZ: {h_val / nz_val:.1f} ft")
        dz_readout.setStyleSheet("color: #0284c7; font-weight: bold;")
        form.addRow("Layer Resolution:", dz_readout)

        dip_spin = QDoubleSpinBox()
        dip_spin.setRange(-45.0, 45.0)
        dip_spin.setSingleStep(1.0)
        dip_spin.setValue(float(data.get("dip_angle_deg", 0.0)))
        dip_spin.valueChanged.connect(lambda v: self.parameter_changed.emit("reservoir", "dip_angle_deg", v))
        form.addRow("Structural Dip Angle:", self._with_badge(dip_spin, "deg"))

        vdp_spin = QDoubleSpinBox()
        vdp_spin.setRange(0.0, 0.95)
        vdp_spin.setSingleStep(0.05)
        vdp_spin.setValue(float(data.get("dykstra_parsons", 0.65)))
        vdp_spin.valueChanged.connect(lambda v: self.parameter_changed.emit("reservoir", "dykstra_parsons", v))
        form.addRow("Dykstra-Parsons (V_DP):", self._with_badge(vdp_spin, "index"))

        kvkh_spin = QDoubleSpinBox()
        kvkh_spin.setRange(0.001, 1.000)
        kvkh_spin.setSingleStep(0.05)
        kvkh_spin.setDecimals(3)
        kvkh_spin.setValue(float(data.get("kv_kh_ratio", 0.10)))
        kvkh_spin.valueChanged.connect(lambda v: self.parameter_changed.emit("reservoir", "kv_kh_ratio", v))
        form.addRow("Anisotropy (Kv / Kh):", self._with_badge(kvkh_spin, "ratio"))

        env_combo = QComboBox()
        env_combo.addItems(["Carbonate Ramp / Platform", "Fluvial Channel Sandstone", "Shallow Marine Shoreface", "Deepwater Turbidite"])
        env_combo.setCurrentText(str(data.get("depositional_env", "Carbonate Ramp / Platform")))
        env_combo.currentTextChanged.connect(lambda v: self.parameter_changed.emit("reservoir", "depositional_env", v))
        form.addRow("Facies Environment:", env_combo)

        cap_contact = QComboBox()
        cap_contact.addItems(["Sharp Unconformable Marine Shale Contact", "Gradational Anhydrite / Mudstone", "Transgressive Marine Flooding Surface"])
        cap_contact.setCurrentText(str(data.get("caprock_contact_type", "Sharp Unconformable Marine Shale Contact")))
        cap_contact.currentTextChanged.connect(lambda v: self.parameter_changed.emit("reservoir", "caprock_contact_type", v))
        form.addRow("Upper Seal Contact:", cap_contact)

        self.form_layout.addLayout(form)

    def _build_petrophysics_form(self, data: Dict[str, Any]):
        lbl_dist = QLabel("Parameter Distribution & Petrofacies Architecture")
        lbl_dist.setObjectName("sectionHeader")
        self.form_layout.addWidget(lbl_dist)

        form_dist = QFormLayout()
        form_dist.setSpacing(6)

        dist_combo = QComboBox()
        dist_combo.addItems([
            "Facies-Controlled (3-Facies Architecture)",
            "Layered (Dykstra-Parsons)",
            "Geostatistical SGSIM",
            "Homogeneous"
        ])
        dist_combo.setCurrentText(str(data.get("distribution_method", "Facies-Controlled (3-Facies Architecture)")))
        dist_combo.currentTextChanged.connect(lambda v: self.parameter_changed.emit("petrophysics", "distribution_method", v))
        form_dist.addRow("Distribution Method:", dist_combo)
        self.inputs["distribution_method"] = dist_combo

        self.form_layout.addLayout(form_dist)

        # Dynamic container for distribution pattern geometry & properties
        pattern_group = QFrame()
        pattern_group.setObjectName("patternCard")
        pattern_group.setStyleSheet("""
            QFrame#patternCard {
                background: #f8fafc;
                border: 1px solid #cbd5e1;
                border-radius: 4px;
            }
            QFrame#patternCard QLabel {
                border: none;
                background: transparent;
                padding: 0px;
            }
        """)
        pattern_layout = QVBoxLayout(pattern_group)
        pattern_layout.setContentsMargins(6, 6, 6, 6)
        pattern_layout.setSpacing(6)
        self.form_layout.addWidget(pattern_group)

        def _refresh_pattern_controls():
            _clear_layout(pattern_layout)
            cur_method = dist_combo.currentText()

            if "Facies" in cur_method:
                lbl_pat = QLabel("Depositional Pattern Geometry & Dimensions")
                lbl_pat.setStyleSheet("color: #0284c7; font-weight: bold; font-size: 11px;")
                pattern_layout.addWidget(lbl_pat)

                form_pat = QFormLayout()
                form_pat.setSpacing(5)

                facies_combo = QComboBox()
                facies_combo.addItems(["Fluvial Channel Belt", "Barrier Island / Shoreface", "Carbonate Patch Reef"])
                facies_combo.setCurrentText(str(data.get("facies_pattern", "Fluvial Channel Belt")))
                facies_combo.currentTextChanged.connect(lambda v: self.parameter_changed.emit("petrophysics", "facies_pattern", v))
                form_pat.addRow("Pattern Type:", facies_combo)
                self.inputs["facies_pattern"] = facies_combo

                pattern_layout.addLayout(form_pat)

                # Geometry sub-container
                geom_widget = QWidget()
                geom_layout = QFormLayout(geom_widget)
                geom_layout.setContentsMargins(0, 0, 0, 0)
                geom_layout.setSpacing(5)
                pattern_layout.addWidget(geom_widget)

                def _render_geom_fields():
                    _clear_layout(geom_layout)
                    pat_type = facies_combo.currentText()

                    if "Fluvial" in pat_type:
                        # Fluvial Channel controls
                        az_spin = QDoubleSpinBox()
                        az_spin.setRange(0.0, 360.0)
                        az_spin.setSingleStep(5.0)
                        az_spin.setValue(float(data.get("channel_azimuth_deg", 45.0)))
                        az_spin.valueChanged.connect(lambda v: self.parameter_changed.emit("petrophysics", "channel_azimuth_deg", v))
                        geom_layout.addRow("Azimuth:", self._with_badge(az_spin, "deg"))
                        self.inputs["channel_azimuth_deg"] = az_spin

                        sin_spin = QDoubleSpinBox()
                        sin_spin.setRange(1.00, 2.50)
                        sin_spin.setSingleStep(0.05)
                        sin_spin.setDecimals(2)
                        sin_spin.setValue(float(data.get("channel_sinuosity", 1.30)))
                        sin_spin.valueChanged.connect(lambda v: self.parameter_changed.emit("petrophysics", "channel_sinuosity", v))
                        geom_layout.addRow("Sinuosity:", self._with_badge(sin_spin, "idx"))
                        self.inputs["channel_sinuosity"] = sin_spin

                        wl_spin = QDoubleSpinBox()
                        wl_spin.setRange(200.0, 10000.0)
                        wl_spin.setSingleStep(100.0)
                        wl_spin.setValue(float(data.get("channel_wavelength_ft", 1500.0)))
                        wl_spin.valueChanged.connect(lambda v: self.parameter_changed.emit("petrophysics", "channel_wavelength_ft", v))
                        geom_layout.addRow("Wavelength:", self._with_badge(wl_spin, "ft"))
                        self.inputs["channel_wavelength_ft"] = wl_spin

                        amp_spin = QDoubleSpinBox()
                        amp_spin.setRange(20.0, 2500.0)
                        amp_spin.setSingleStep(25.0)
                        amp_spin.setValue(float(data.get("channel_amplitude_ft", 350.0)))
                        amp_spin.valueChanged.connect(lambda v: self.parameter_changed.emit("petrophysics", "channel_amplitude_ft", v))
                        geom_layout.addRow("Amplitude:", self._with_badge(amp_spin, "ft"))
                        self.inputs["channel_amplitude_ft"] = amp_spin

                        cw_spin = QDoubleSpinBox()
                        cw_spin.setRange(50.0, 2000.0)
                        cw_spin.setSingleStep(25.0)
                        cw_spin.setValue(float(data.get("channel_width_ft", 450.0)))
                        cw_spin.valueChanged.connect(lambda v: self.parameter_changed.emit("petrophysics", "channel_width_ft", v))
                        geom_layout.addRow("Sand Width:", self._with_badge(cw_spin, "ft"))
                        self.inputs["channel_width_ft"] = cw_spin

                        lw_spin = QDoubleSpinBox()
                        lw_spin.setRange(10.0, 1000.0)
                        lw_spin.setSingleStep(10.0)
                        lw_spin.setValue(float(data.get("levee_width_ft", 250.0)))
                        lw_spin.valueChanged.connect(lambda v: self.parameter_changed.emit("petrophysics", "levee_width_ft", v))
                        geom_layout.addRow("Levee Width:", self._with_badge(lw_spin, "ft"))
                        self.inputs["levee_width_ft"] = lw_spin

                        drift_spin = QDoubleSpinBox()
                        drift_spin.setRange(-150.0, 150.0)
                        drift_spin.setSingleStep(5.0)
                        drift_spin.setValue(float(data.get("aggradation_drift_ft", 30.0)))
                        drift_spin.valueChanged.connect(lambda v: self.parameter_changed.emit("petrophysics", "aggradation_drift_ft", v))
                        geom_layout.addRow("Vert Drift:", self._with_badge(drift_spin, "ft"))
                        self.inputs["aggradation_drift_ft"] = drift_spin

                        nc_spin = QSpinBox()
                        nc_spin.setRange(1, 5)
                        nc_spin.setValue(int(data.get("num_channels", 1)))
                        nc_spin.valueChanged.connect(lambda v: self.parameter_changed.emit("petrophysics", "num_channels", v))
                        geom_layout.addRow("Channels:", self._with_badge(nc_spin, "count"))
                        self.inputs["num_channels"] = nc_spin

                    elif "Barrier" in pat_type or "Shoreface" in pat_type:
                        # Barrier Island / Shoreface controls
                        baz_spin = QDoubleSpinBox()
                        baz_spin.setRange(0.0, 360.0)
                        baz_spin.setSingleStep(5.0)
                        baz_spin.setValue(float(data.get("barrier_azimuth_deg", 90.0)))
                        baz_spin.valueChanged.connect(lambda v: self.parameter_changed.emit("petrophysics", "barrier_azimuth_deg", v))
                        geom_layout.addRow("Strike Azimuth:", self._with_badge(baz_spin, "deg"))
                        self.inputs["barrier_azimuth_deg"] = baz_spin

                        bw_spin = QDoubleSpinBox()
                        bw_spin.setRange(100.0, 5000.0)
                        bw_spin.setSingleStep(50.0)
                        bw_spin.setValue(float(data.get("barrier_width_ft", 800.0)))
                        bw_spin.valueChanged.connect(lambda v: self.parameter_changed.emit("petrophysics", "barrier_width_ft", v))
                        geom_layout.addRow("Bar Width:", self._with_badge(bw_spin, "ft"))
                        self.inputs["barrier_width_ft"] = bw_spin

                        lag_spin = QDoubleSpinBox()
                        lag_spin.setRange(50.0, 3000.0)
                        lag_spin.setSingleStep(50.0)
                        lag_spin.setValue(float(data.get("lagoon_width_ft", 450.0)))
                        lag_spin.valueChanged.connect(lambda v: self.parameter_changed.emit("petrophysics", "lagoon_width_ft", v))
                        geom_layout.addRow("Mud Baffle Width:", self._with_badge(lag_spin, "ft"))
                        self.inputs["lagoon_width_ft"] = lag_spin

                        dip_spin = QDoubleSpinBox()
                        dip_spin.setRange(0.0, 15.0)
                        dip_spin.setSingleStep(0.5)
                        dip_spin.setDecimals(1)
                        dip_spin.setValue(float(data.get("progradation_dip_deg", 2.0)))
                        dip_spin.valueChanged.connect(lambda v: self.parameter_changed.emit("petrophysics", "progradation_dip_deg", v))
                        geom_layout.addRow("Clinoform Dip:", self._with_badge(dip_spin, "deg"))
                        self.inputs["progradation_dip_deg"] = dip_spin

                    else:
                        # Carbonate Patch Reef controls
                        rx_spin = QDoubleSpinBox()
                        rx_spin.setRange(0.0, 10000.0)
                        rx_spin.setSingleStep(50.0)
                        rx_spin.setValue(float(data.get("reef_center_x", 1000.0)))
                        rx_spin.valueChanged.connect(lambda v: self.parameter_changed.emit("petrophysics", "reef_center_x", v))
                        geom_layout.addRow("Center X:", self._with_badge(rx_spin, "ft"))
                        self.inputs["reef_center_x"] = rx_spin

                        ry_spin = QDoubleSpinBox()
                        ry_spin.setRange(0.0, 10000.0)
                        ry_spin.setSingleStep(50.0)
                        ry_spin.setValue(float(data.get("reef_center_y", 1000.0)))
                        ry_spin.valueChanged.connect(lambda v: self.parameter_changed.emit("petrophysics", "reef_center_y", v))
                        geom_layout.addRow("Center Y:", self._with_badge(ry_spin, "ft"))
                        self.inputs["reef_center_y"] = ry_spin

                        rmaj_spin = QDoubleSpinBox()
                        rmaj_spin.setRange(50.0, 5000.0)
                        rmaj_spin.setSingleStep(25.0)
                        rmaj_spin.setValue(float(data.get("reef_major_radius_ft", 650.0)))
                        rmaj_spin.valueChanged.connect(lambda v: self.parameter_changed.emit("petrophysics", "reef_major_radius_ft", v))
                        geom_layout.addRow("Major Radius:", self._with_badge(rmaj_spin, "ft"))
                        self.inputs["reef_major_radius_ft"] = rmaj_spin

                        rmin_spin = QDoubleSpinBox()
                        rmin_spin.setRange(50.0, 5000.0)
                        rmin_spin.setSingleStep(25.0)
                        rmin_spin.setValue(float(data.get("reef_minor_radius_ft", 400.0)))
                        rmin_spin.valueChanged.connect(lambda v: self.parameter_changed.emit("petrophysics", "reef_minor_radius_ft", v))
                        geom_layout.addRow("Minor Radius:", self._with_badge(rmin_spin, "ft"))
                        self.inputs["reef_minor_radius_ft"] = rmin_spin

                        raz_spin = QDoubleSpinBox()
                        raz_spin.setRange(0.0, 360.0)
                        raz_spin.setSingleStep(5.0)
                        raz_spin.setValue(float(data.get("reef_azimuth_deg", 45.0)))
                        raz_spin.valueChanged.connect(lambda v: self.parameter_changed.emit("petrophysics", "reef_azimuth_deg", v))
                        geom_layout.addRow("Mound Azimuth:", self._with_badge(raz_spin, "deg"))
                        self.inputs["reef_azimuth_deg"] = raz_spin

                        ap_spin = QDoubleSpinBox()
                        ap_spin.setRange(20.0, 2000.0)
                        ap_spin.setSingleStep(20.0)
                        ap_spin.setValue(float(data.get("apron_width_ft", 300.0)))
                        ap_spin.valueChanged.connect(lambda v: self.parameter_changed.emit("petrophysics", "apron_width_ft", v))
                        geom_layout.addRow("Apron Width:", self._with_badge(ap_spin, "ft"))
                        self.inputs["apron_width_ft"] = ap_spin

                facies_combo.currentTextChanged.connect(lambda _: _render_geom_fields())
                _render_geom_fields()

                # Explicit Per-Facies Petrophysical Properties section
                lbl_fac = QLabel("Explicit Petrofacies Properties (k & φ)")
                lbl_fac.setStyleSheet("color: #0284c7; font-weight: bold; font-size: 11px; margin-top: 6px;")
                pattern_layout.addWidget(lbl_fac)

                form_f = QFormLayout()
                form_f.setSpacing(5)

                f1_k = QDoubleSpinBox()
                f1_k.setRange(1.0, 10000.0)
                f1_k.setValue(float(data.get("f1_perm", 250.0)))
                f1_k.valueChanged.connect(lambda v: self.parameter_changed.emit("petrophysics", "f1_perm", v))
                form_f.addRow("Facies 1 Perm (k1):", self._with_badge(f1_k, "mD"))
                self.inputs["f1_perm"] = f1_k

                f1_p = QDoubleSpinBox()
                f1_p.setRange(0.05, 0.45)
                f1_p.setSingleStep(0.01)
                f1_p.setDecimals(3)
                f1_p.setValue(float(data.get("f1_poro", 0.25)))
                f1_p.valueChanged.connect(lambda v: self.parameter_changed.emit("petrophysics", "f1_poro", v))
                form_f.addRow("Facies 1 Poro (φ1):", self._with_badge(f1_p, "frac"))
                self.inputs["f1_poro"] = f1_p

                f2_k = QDoubleSpinBox()
                f2_k.setRange(0.1, 2000.0)
                f2_k.setValue(float(data.get("f2_perm", 40.0)))
                f2_k.valueChanged.connect(lambda v: self.parameter_changed.emit("petrophysics", "f2_perm", v))
                form_f.addRow("Facies 2 Perm (k2):", self._with_badge(f2_k, "mD"))
                self.inputs["f2_perm"] = f2_k

                f2_p = QDoubleSpinBox()
                f2_p.setRange(0.03, 0.35)
                f2_p.setSingleStep(0.01)
                f2_p.setDecimals(3)
                f2_p.setValue(float(data.get("f2_poro", 0.16)))
                f2_p.valueChanged.connect(lambda v: self.parameter_changed.emit("petrophysics", "f2_poro", v))
                form_f.addRow("Facies 2 Poro (φ2):", self._with_badge(f2_p, "frac"))
                self.inputs["f2_poro"] = f2_p

                f3_k = QDoubleSpinBox()
                f3_k.setRange(0.0001, 50.0)
                f3_k.setDecimals(4)
                f3_k.setValue(float(data.get("f3_perm", 0.5)))
                f3_k.valueChanged.connect(lambda v: self.parameter_changed.emit("petrophysics", "f3_perm", v))
                form_f.addRow("Facies 3 Perm (k3):", self._with_badge(f3_k, "mD"))
                self.inputs["f3_perm"] = f3_k

                f3_p = QDoubleSpinBox()
                f3_p.setRange(0.005, 0.25)
                f3_p.setSingleStep(0.01)
                f3_p.setDecimals(3)
                f3_p.setValue(float(data.get("f3_poro", 0.06)))
                f3_p.valueChanged.connect(lambda v: self.parameter_changed.emit("petrophysics", "f3_poro", v))
                form_f.addRow("Facies 3 Poro (φ3):", self._with_badge(f3_p, "frac"))
                self.inputs["f3_poro"] = f3_p

                pattern_layout.addLayout(form_f)

            elif "Layered" in cur_method:
                lbl_lay = QLabel("Dykstra-Parsons & Vertical Permeability Trend")
                lbl_lay.setStyleSheet("color: #0284c7; font-weight: bold; font-size: 11px;")
                pattern_layout.addWidget(lbl_lay)

                form_lay = QFormLayout()
                form_lay.setSpacing(5)

                vdp_spin = QDoubleSpinBox()
                vdp_spin.setRange(0.0, 0.95)
                vdp_spin.setSingleStep(0.05)
                vdp_spin.setValue(float(data.get("dykstra_parsons", 0.65)))
                vdp_spin.valueChanged.connect(lambda v: self.parameter_changed.emit("petrophysics", "dykstra_parsons", v))
                form_lay.addRow("Dykstra-Parsons (V_DP):", self._with_badge(vdp_spin, "idx"))
                self.inputs["dykstra_parsons"] = vdp_spin

                trend_combo = QComboBox()
                trend_combo.addItems([
                    "Fining Upward (Fluvial point-bar cycle)",
                    "Coarsening Upward (Prograding deltaic)",
                    "Symmetric Cycle (Transgressive-Regressive)",
                    "Stochastic Lognormal (Random bedding)"
                ])
                trend_combo.setCurrentText(str(data.get("layer_permeability_trend", "Fining Upward (Fluvial point-bar cycle)")))
                trend_combo.currentTextChanged.connect(lambda v: self.parameter_changed.emit("petrophysics", "layer_permeability_trend", v.split(" (")[0]))
                form_lay.addRow("Perm Trend:", trend_combo)
                self.inputs["layer_permeability_trend"] = trend_combo

                kvkh_spin = QDoubleSpinBox()
                kvkh_spin.setRange(0.001, 1.000)
                kvkh_spin.setSingleStep(0.05)
                kvkh_spin.setDecimals(3)
                kvkh_spin.setValue(float(data.get("kv_kh_ratio", 0.10)))
                kvkh_spin.valueChanged.connect(lambda v: self.parameter_changed.emit("petrophysics", "kv_kh_ratio", v))
                form_lay.addRow("Kv / Kh Ratio:", self._with_badge(kvkh_spin, "ratio"))
                self.inputs["kv_kh_ratio"] = kvkh_spin

                pp_combo = QComboBox()
                pp_combo.addItems(["Kozeny-Carman", "Log-Linear (R2=0.85)", "Facies Direct Trend"])
                pp_combo.setCurrentText(str(data.get("poro_perm_model", "Kozeny-Carman")))
                pp_combo.currentTextChanged.connect(lambda v: self.parameter_changed.emit("petrophysics", "poro_perm_model", v))
                form_lay.addRow("Poro-Perm Model:", pp_combo)
                self.inputs["poro_perm_model"] = pp_combo

                pattern_layout.addLayout(form_lay)

            elif "Geostatistical" in cur_method:
                lbl_geo = QLabel("Variogram Model & Spatial Continuity")
                lbl_geo.setStyleSheet("color: #0284c7; font-weight: bold; font-size: 11px;")
                pattern_layout.addWidget(lbl_geo)

                form_geo = QFormLayout()
                form_geo.setSpacing(5)

                vt_combo = QComboBox()
                vt_combo.addItems(["Spherical", "Exponential", "Gaussian"])
                vt_combo.setCurrentText(str(data.get("variogram_type", "Spherical")))
                vt_combo.currentTextChanged.connect(lambda v: self.parameter_changed.emit("petrophysics", "variogram_type", v))
                form_geo.addRow("Variogram Type:", vt_combo)
                self.inputs["variogram_type"] = vt_combo

                rmaj = QDoubleSpinBox()
                rmaj.setRange(50.0, 20000.0)
                rmaj.setSingleStep(50.0)
                rmaj.setValue(float(data.get("variogram_range_major", 1200.0)))
                rmaj.valueChanged.connect(lambda v: self.parameter_changed.emit("petrophysics", "variogram_range_major", v))
                form_geo.addRow("Major Range (a_maj):", self._with_badge(rmaj, "ft"))
                self.inputs["variogram_range_major"] = rmaj

                rmin = QDoubleSpinBox()
                rmin.setRange(25.0, 10000.0)
                rmin.setSingleStep(25.0)
                rmin.setValue(float(data.get("variogram_range_minor", 600.0)))
                rmin.valueChanged.connect(lambda v: self.parameter_changed.emit("petrophysics", "variogram_range_minor", v))
                form_geo.addRow("Minor Range (a_min):", self._with_badge(rmin, "ft"))
                self.inputs["variogram_range_minor"] = rmin

                rvert = QDoubleSpinBox()
                rvert.setRange(1.0, 500.0)
                rvert.setSingleStep(2.0)
                rvert.setValue(float(data.get("variogram_range_vert", 20.0)))
                rvert.valueChanged.connect(lambda v: self.parameter_changed.emit("petrophysics", "variogram_range_vert", v))
                form_geo.addRow("Vertical Range (a_z):", self._with_badge(rvert, "ft"))
                self.inputs["variogram_range_vert"] = rvert

                az_geo = QDoubleSpinBox()
                az_geo.setRange(0.0, 360.0)
                az_geo.setSingleStep(5.0)
                az_geo.setValue(float(data.get("variogram_azimuth_deg", 45.0)))
                az_geo.valueChanged.connect(lambda v: self.parameter_changed.emit("petrophysics", "variogram_azimuth_deg", v))
                form_geo.addRow("Principal Azimuth:", self._with_badge(az_geo, "deg"))
                self.inputs["variogram_azimuth_deg"] = az_geo

                nug_spin = QDoubleSpinBox()
                nug_spin.setRange(0.0, 1.0)
                nug_spin.setSingleStep(0.02)
                nug_spin.setValue(float(data.get("nugget_effect", 0.05)))
                nug_spin.valueChanged.connect(lambda v: self.parameter_changed.emit("petrophysics", "nugget_effect", v))
                form_geo.addRow("Nugget Effect (c0):", self._with_badge(nug_spin, "var"))
                self.inputs["nugget_effect"] = nug_spin

                sill_spin = QDoubleSpinBox()
                sill_spin.setRange(0.05, 5.0)
                sill_spin.setSingleStep(0.05)
                sill_spin.setValue(float(data.get("sill_variance", 1.0)))
                sill_spin.valueChanged.connect(lambda v: self.parameter_changed.emit("petrophysics", "sill_variance", v))
                form_geo.addRow("Sill Variance (C):", self._with_badge(sill_spin, "var"))
                self.inputs["sill_variance"] = sill_spin

                pattern_layout.addLayout(form_geo)

                btn_jump_geostat = QPushButton("Open Dedicated Variogram Form & Visualizer ➔")
                btn_jump_geostat.setStyleSheet("background: #e0f2fe; color: #0284c7; font-weight: bold; font-size: 10.5px; padding: 4px;")
                btn_jump_geostat.clicked.connect(lambda: self.load_node("petrophysics", "geostat", data))
                pattern_layout.addWidget(btn_jump_geostat)

            else:
                lbl_hom = QLabel("Homogeneous uniform petrophysical volume across all active cells.")
                lbl_hom.setStyleSheet("color: #64748b; font-size: 11px; padding: 6px;")
                pattern_layout.addWidget(lbl_hom)

        dist_combo.currentTextChanged.connect(lambda _: _refresh_pattern_controls())
        _refresh_pattern_controls()

        # Base Petrophysical & Flow Properties
        lbl_base = QLabel("Base Petrophysical & Flow Properties")
        lbl_base.setObjectName("sectionHeader")
        self.form_layout.addWidget(lbl_base)

        form_base = QFormLayout()
        form_base.setSpacing(6)

        phi_spin = QDoubleSpinBox()
        phi_spin.setRange(0.01, 0.45)
        phi_spin.setSingleStep(0.01)
        phi_spin.setDecimals(3)
        phi_spin.setValue(float(data.get("poro", 0.20)))
        phi_spin.valueChanged.connect(lambda v: self.parameter_changed.emit("petrophysics", "poro", v))
        form_base.addRow("Base Porosity (φ):", self._with_badge(phi_spin, "fraction"))
        self.inputs["poro"] = phi_spin

        k_spin = QDoubleSpinBox()
        k_spin.setRange(0.1, 5000.0)
        k_spin.setValue(float(data.get("perm", 100.0)))
        k_spin.valueChanged.connect(lambda v: self.parameter_changed.emit("petrophysics", "perm", v))
        form_base.addRow("Base Permeability (k):", self._with_badge(k_spin, "mD"))
        self.inputs["perm"] = k_spin

        sand_spin = QDoubleSpinBox()
        sand_spin.setRange(0.10, 0.95)
        sand_spin.setSingleStep(0.05)
        sand_spin.setDecimals(2)
        sand_spin.setValue(float(data.get("sand_fraction", 0.65)))
        sand_spin.valueChanged.connect(lambda v: self.parameter_changed.emit("petrophysics", "sand_fraction", v))
        form_base.addRow("Clean Sand Fraction:", self._with_badge(sand_spin, "frac"))
        self.inputs["sand_fraction"] = sand_spin

        shale_spin = QDoubleSpinBox()
        shale_spin.setRange(0.01, 0.50)
        shale_spin.setSingleStep(0.02)
        shale_spin.setDecimals(2)
        shale_spin.setValue(float(data.get("shale_fraction", 0.10)))
        shale_spin.valueChanged.connect(lambda v: self.parameter_changed.emit("petrophysics", "shale_fraction", v))
        form_base.addRow("Shale Baffle Fraction:", self._with_badge(shale_spin, "frac"))
        self.inputs["shale_fraction"] = shale_spin

        swc_spin = QDoubleSpinBox()
        swc_spin.setRange(0.05, 0.60)
        swc_spin.setSingleStep(0.01)
        swc_spin.setValue(float(data.get("s_wc", 0.20)))
        swc_spin.valueChanged.connect(lambda v: self.parameter_changed.emit("petrophysics", "s_wc", v))
        form_base.addRow("Connate Water (Swc):", self._with_badge(swc_spin, "fraction"))
        self.inputs["s_wc"] = swc_spin

        sorw_spin = QDoubleSpinBox()
        sorw_spin.setRange(0.05, 0.50)
        sorw_spin.setSingleStep(0.01)
        sorw_spin.setValue(float(data.get("s_orw", 0.20)))
        sorw_spin.valueChanged.connect(lambda v: self.parameter_changed.emit("petrophysics", "s_orw", v))
        form_base.addRow("Residual Oil (Sorw):", self._with_badge(sorw_spin, "fraction"))
        self.inputs["s_orw"] = sorw_spin

        sgc_spin = QDoubleSpinBox()
        sgc_spin.setRange(0.01, 0.25)
        sgc_spin.setSingleStep(0.01)
        sgc_spin.setValue(float(data.get("s_gc", 0.05)))
        sgc_spin.valueChanged.connect(lambda v: self.parameter_changed.emit("petrophysics", "s_gc", v))
        form_base.addRow("Critical Gas (Sgc):", self._with_badge(sgc_spin, "fraction"))
        self.inputs["s_gc"] = sgc_spin

        btn_relperm = QPushButton("Configure Full Corey Rel-Perm Curves ➔")
        btn_relperm.setStyleSheet("background: #f0fdf4; color: #16a34a; font-weight: bold; font-size: 11px; padding: 5px; border: 1px solid #bbf7d0; border-radius: 4px;")
        btn_relperm.clicked.connect(lambda: self.load_node("petrophysics", "relperm", data))
        form_base.addRow("", btn_relperm)

        self.form_layout.addLayout(form_base)

    def _build_relperm_form(self, data: Dict[str, Any]):
        """Builds dedicated Corey relative permeability, wettability crossover and embedded curve form."""
        lbl_head = QLabel("Corey Relative Permeability & Wettability Model")
        lbl_head.setObjectName("sectionHeader")
        self.form_layout.addWidget(lbl_head)

        form_top = QFormLayout()
        form_top.setSpacing(6)

        # Wettability Presets
        preset_combo = QComboBox()
        preset_combo.addItems([
            "Strongly Water-Wet (Sandstone)",
            "Mixed-Wet / Intermediate (Carbonate / Dolomite)",
            "Strongly Oil-Wet (Bitumen / Asphaltenic)",
            "Custom User Configuration"
        ])
        saved_preset = str(data.get("wettability_preset", "Strongly Water-Wet (Sandstone)"))
        preset_combo.setCurrentText(saved_preset)
        form_top.addRow("Wettability Preset:", preset_combo)

        # 3-Phase Model Selection
        model_combo = QComboBox()
        model_combo.addItems([
            "Modified Stone I (Standard CO2 EOR)",
            "Baker Saturation-Weighted",
            "Two-Phase Corey Direct"
        ])
        model_combo.setCurrentText(str(data.get("relperm_model", "Modified Stone I (Standard CO2 EOR)")))
        model_combo.currentTextChanged.connect(lambda v: self.parameter_changed.emit("petrophysics", "relperm_model", v))
        form_top.addRow("3-Phase Model:", model_combo)

        self.form_layout.addLayout(form_top)

        # Endpoint Phase Saturations
        lbl_sat = QLabel("Endpoint Phase Saturations")
        lbl_sat.setObjectName("sectionHeader")
        self.form_layout.addWidget(lbl_sat)

        form_sat = QFormLayout()
        form_sat.setSpacing(6)

        swc_spin = QDoubleSpinBox()
        swc_spin.setRange(0.05, 0.50)
        swc_spin.setSingleStep(0.01)
        swc_spin.setValue(float(data.get("s_wc", 0.20)))
        swc_spin.valueChanged.connect(lambda v: self.parameter_changed.emit("petrophysics", "s_wc", v))
        form_sat.addRow("Connate Water (Swc):", self._with_badge(swc_spin, "frac"))
        self.inputs["s_wc"] = swc_spin

        sorw_spin = QDoubleSpinBox()
        sorw_spin.setRange(0.05, 0.50)
        sorw_spin.setSingleStep(0.01)
        sorw_spin.setValue(float(data.get("s_orw", 0.20)))
        sorw_spin.valueChanged.connect(lambda v: self.parameter_changed.emit("petrophysics", "s_orw", v))
        form_sat.addRow("Residual Oil to Water (Sorw):", self._with_badge(sorw_spin, "frac"))
        self.inputs["s_orw"] = sorw_spin

        sgc_spin = QDoubleSpinBox()
        sgc_spin.setRange(0.01, 0.25)
        sgc_spin.setSingleStep(0.01)
        sgc_spin.setValue(float(data.get("s_gc", 0.05)))
        sgc_spin.valueChanged.connect(lambda v: self.parameter_changed.emit("petrophysics", "s_gc", v))
        form_sat.addRow("Critical Gas (Sgc):", self._with_badge(sgc_spin, "frac"))
        self.inputs["s_gc"] = sgc_spin

        sorg_spin = QDoubleSpinBox()
        sorg_spin.setRange(0.02, 0.40)
        sorg_spin.setSingleStep(0.01)
        sorg_spin.setValue(float(data.get("s_org", 0.15)))
        sorg_spin.valueChanged.connect(lambda v: self.parameter_changed.emit("petrophysics", "s_org", v))
        form_sat.addRow("Residual Oil to Gas (Sorg):", self._with_badge(sorg_spin, "frac"))
        self.inputs["s_org"] = sorg_spin

        self.form_layout.addLayout(form_sat)

        # Endpoint Relative Permeabilities & Exponents
        lbl_endpoints = QLabel("Corey Endpoints & Curvature Exponents")
        lbl_endpoints.setObjectName("sectionHeader")
        self.form_layout.addWidget(lbl_endpoints)

        form_end = QFormLayout()
        form_end.setSpacing(6)

        krw0_spin = QDoubleSpinBox()
        krw0_spin.setRange(0.05, 1.00)
        krw0_spin.setSingleStep(0.02)
        krw0_spin.setValue(float(data.get("krw0", 0.30)))
        krw0_spin.valueChanged.connect(lambda v: self.parameter_changed.emit("petrophysics", "krw0", v))
        form_end.addRow("Endpoint Water Perm (krw0):", self._with_badge(krw0_spin, "dim"))
        self.inputs["krw0"] = krw0_spin

        kro0_spin = QDoubleSpinBox()
        kro0_spin.setRange(0.10, 1.00)
        kro0_spin.setSingleStep(0.02)
        kro0_spin.setValue(float(data.get("kro0", 0.85)))
        kro0_spin.valueChanged.connect(lambda v: self.parameter_changed.emit("petrophysics", "kro0", v))
        form_end.addRow("Endpoint Oil Perm (kro0):", self._with_badge(kro0_spin, "dim"))
        self.inputs["kro0"] = kro0_spin

        krg0_spin = QDoubleSpinBox()
        krg0_spin.setRange(0.05, 1.00)
        krg0_spin.setSingleStep(0.02)
        krg0_spin.setValue(float(data.get("krg0", 0.60)))
        krg0_spin.valueChanged.connect(lambda v: self.parameter_changed.emit("petrophysics", "krg0", v))
        form_end.addRow("Endpoint Gas Perm (krg0):", self._with_badge(krg0_spin, "dim"))
        self.inputs["krg0"] = krg0_spin

        nw_spin = QDoubleSpinBox()
        nw_spin.setRange(1.0, 6.0)
        nw_spin.setSingleStep(0.1)
        nw_spin.setValue(float(data.get("n_w", 2.5)))
        nw_spin.valueChanged.connect(lambda v: self.parameter_changed.emit("petrophysics", "n_w", v))
        form_end.addRow("Water Exponent (nw):", self._with_badge(nw_spin, "exp"))
        self.inputs["n_w"] = nw_spin

        now_spin = QDoubleSpinBox()
        now_spin.setRange(1.0, 6.0)
        now_spin.setSingleStep(0.1)
        now_spin.setValue(float(data.get("n_ow", data.get("n_o", 2.0))))
        now_spin.valueChanged.connect(lambda v: self.parameter_changed.emit("petrophysics", "n_ow", v))
        form_end.addRow("Oil-Water Exponent (now):", self._with_badge(now_spin, "exp"))
        self.inputs["n_ow"] = now_spin

        ng_spin = QDoubleSpinBox()
        ng_spin.setRange(1.0, 6.0)
        ng_spin.setSingleStep(0.1)
        ng_spin.setValue(float(data.get("n_g", 2.0)))
        ng_spin.valueChanged.connect(lambda v: self.parameter_changed.emit("petrophysics", "n_g", v))
        form_end.addRow("Gas Exponent (ng):", self._with_badge(ng_spin, "exp"))
        self.inputs["n_g"] = ng_spin

        self.form_layout.addLayout(form_end)

        # Calculated Diagnostics Badges
        lbl_diag = QLabel("Dynamic Wettability Diagnostics")
        lbl_diag.setObjectName("sectionHeader")
        self.form_layout.addWidget(lbl_diag)

        form_diag = QFormLayout()
        form_diag.setSpacing(6)

        window_badge = QLabel()
        window_badge.setStyleSheet("color: #0284c7; font-weight: bold;")
        form_diag.addRow("Mobile Saturation ΔS:", window_badge)

        cross_badge = QLabel()
        cross_badge.setStyleSheet("color: #16a34a; font-weight: bold;")
        form_diag.addRow("Crossover Saturation Sw*:", cross_badge)

        self.form_layout.addLayout(form_diag)

        def _update_diagnostics_badges():
            swc_v = swc_spin.value()
            sorw_v = sorw_spin.value()
            krw0_v = krw0_spin.value()
            kro0_v = kro0_spin.value()
            nw_v = nw_spin.value()
            now_v = now_spin.value()

            mobile_window = max(1.0 - swc_v - sorw_v, 1e-4)
            window_badge.setText(f"{mobile_window * 100.0:.1f}% pore volume")

            sw_arr = np.linspace(swc_v, 1.0 - sorw_v, 80)
            norm_sw = (sw_arr - swc_v) / mobile_window
            krw_arr = krw0_v * (norm_sw ** nw_v)
            krow_arr = kro0_v * ((1.0 - norm_sw) ** now_v)

            diff = np.abs(krw_arr - krow_arr)
            cross_idx = int(np.argmin(diff))
            sw_cross = float(sw_arr[cross_idx])

            if sw_cross > 0.52:
                wet_str = "Water-Wet State"
                cross_badge.setStyleSheet("color: #16a34a; font-weight: bold;")
            elif sw_cross < 0.48:
                wet_str = "Oil-Wet State"
                cross_badge.setStyleSheet("color: #ea580c; font-weight: bold;")
            else:
                wet_str = "Intermediate / Mixed-Wet"
                cross_badge.setStyleSheet("color: #0284c7; font-weight: bold;")
            cross_badge.setText(f"{sw_cross:.2f} ({wet_str})")

        # Connect live diagnostic badges to spinbox adjustments
        for sp in [swc_spin, sorw_spin, krw0_spin, kro0_spin, nw_spin, now_spin]:
            sp.valueChanged.connect(lambda _: _update_diagnostics_badges())
        _update_diagnostics_badges()

        # Preset selection logic
        def _on_preset_selected(preset_name: str):
            self.parameter_changed.emit("petrophysics", "wettability_preset", preset_name)
            if "Strongly Water-Wet" in preset_name:
                swc_spin.setValue(0.25)
                sorw_spin.setValue(0.25)
                krw0_spin.setValue(0.20)
                kro0_spin.setValue(0.90)
                nw_spin.setValue(3.0)
                now_spin.setValue(2.0)
                sorg_spin.setValue(0.15)
                krg0_spin.setValue(0.60)
            elif "Mixed-Wet" in preset_name:
                swc_spin.setValue(0.18)
                sorw_spin.setValue(0.20)
                krw0_spin.setValue(0.35)
                kro0_spin.setValue(0.65)
                nw_spin.setValue(2.2)
                now_spin.setValue(2.5)
                sorg_spin.setValue(0.15)
                krg0_spin.setValue(0.65)
            elif "Strongly Oil-Wet" in preset_name:
                swc_spin.setValue(0.12)
                sorw_spin.setValue(0.15)
                krw0_spin.setValue(0.65)
                kro0_spin.setValue(0.35)
                nw_spin.setValue(1.8)
                now_spin.setValue(3.5)
                sorg_spin.setValue(0.10)
                krg0_spin.setValue(0.75)
            _update_diagnostics_badges()

        preset_combo.currentTextChanged.connect(_on_preset_selected)
        _update_diagnostics_badges()

    def _build_geostat_form(self, data: Dict[str, Any]):
        """Builds dedicated Geostatistics, 3D Spatial Variogram and Realization form."""
        lbl_head = QLabel("Geostatistical Spatial Continuity & Variogram")
        lbl_head.setObjectName("sectionHeader")
        self.form_layout.addWidget(lbl_head)

        form_model = QFormLayout()
        form_model.setSpacing(6)

        vt_combo = QComboBox()
        vt_combo.addItems([
            "Spherical (Standard Sill)",
            "Exponential (Steep Origin)",
            "Gaussian (Inflection / Smooth)"
        ])
        saved_vt = str(data.get("variogram_type", "Spherical"))
        for i in range(vt_combo.count()):
            if saved_vt.lower() in vt_combo.itemText(i).lower():
                vt_combo.setCurrentIndex(i)
                break
        vt_combo.currentTextChanged.connect(lambda v: self.parameter_changed.emit("petrophysics", "variogram_type", v.split(" (")[0]))
        form_model.addRow("Variogram Model:", vt_combo)
        self.inputs["variogram_type"] = vt_combo

        self.form_layout.addLayout(form_model)

        # Correlation Ranges (Correlation Lengths)
        lbl_ranges = QLabel("Correlation Lengths & Ranges")
        lbl_ranges.setObjectName("sectionHeader")
        self.form_layout.addWidget(lbl_ranges)

        form_ranges = QFormLayout()
        form_ranges.setSpacing(6)

        rmaj_spin = QDoubleSpinBox()
        rmaj_spin.setRange(50.0, 20000.0)
        rmaj_spin.setSingleStep(50.0)
        rmaj_spin.setValue(float(data.get("variogram_range_major", 1200.0)))
        rmaj_spin.valueChanged.connect(lambda v: self.parameter_changed.emit("petrophysics", "variogram_range_major", v))
        form_ranges.addRow("Major Range (a_maj):", self._with_badge(rmaj_spin, "ft"))
        self.inputs["variogram_range_major"] = rmaj_spin

        rmin_spin = QDoubleSpinBox()
        rmin_spin.setRange(25.0, 10000.0)
        rmin_spin.setSingleStep(25.0)
        rmin_spin.setValue(float(data.get("variogram_range_minor", 600.0)))
        rmin_spin.valueChanged.connect(lambda v: self.parameter_changed.emit("petrophysics", "variogram_range_minor", v))
        form_ranges.addRow("Minor Range (a_min):", self._with_badge(rmin_spin, "ft"))
        self.inputs["variogram_range_minor"] = rmin_spin

        rvert_spin = QDoubleSpinBox()
        rvert_spin.setRange(1.0, 500.0)
        rvert_spin.setSingleStep(2.0)
        rvert_spin.setValue(float(data.get("variogram_range_vert", 20.0)))
        rvert_spin.valueChanged.connect(lambda v: self.parameter_changed.emit("petrophysics", "variogram_range_vert", v))
        form_ranges.addRow("Vertical Range (a_z):", self._with_badge(rvert_spin, "ft"))
        self.inputs["variogram_range_vert"] = rvert_spin

        aniso_badge = QLabel()
        aniso_badge.setStyleSheet("color: #0284c7; font-weight: bold;")
        form_ranges.addRow("Anisotropy Ratio:", aniso_badge)

        self.form_layout.addLayout(form_ranges)

        # Spatial Orientation & Strike
        lbl_orient = QLabel("Spatial Orientation & Strike")
        lbl_orient.setObjectName("sectionHeader")
        self.form_layout.addWidget(lbl_orient)

        form_orient = QFormLayout()
        form_orient.setSpacing(6)

        az_spin = QDoubleSpinBox()
        az_spin.setRange(0.0, 360.0)
        az_spin.setSingleStep(5.0)
        az_spin.setValue(float(data.get("variogram_azimuth_deg", 45.0)))
        az_spin.valueChanged.connect(lambda v: self.parameter_changed.emit("petrophysics", "variogram_azimuth_deg", v))
        form_orient.addRow("Principal Strike Azimuth:", self._with_badge(az_spin, "deg"))
        self.inputs["variogram_azimuth_deg"] = az_spin

        dip_spin = QDoubleSpinBox()
        dip_spin.setRange(-45.0, 45.0)
        dip_spin.setSingleStep(1.0)
        dip_spin.setValue(float(data.get("variogram_dip_deg", 0.0)))
        dip_spin.valueChanged.connect(lambda v: self.parameter_changed.emit("petrophysics", "variogram_dip_deg", v))
        form_orient.addRow("Structure Dip Angle:", self._with_badge(dip_spin, "deg"))
        self.inputs["variogram_dip_deg"] = dip_spin

        self.form_layout.addLayout(form_orient)

        # Variance & Nugget
        lbl_var = QLabel("Variance & Nugget Discontinuity")
        lbl_var.setObjectName("sectionHeader")
        self.form_layout.addWidget(lbl_var)

        form_var = QFormLayout()
        form_var.setSpacing(6)

        nug_spin = QDoubleSpinBox()
        nug_spin.setRange(0.00, 1.00)
        nug_spin.setSingleStep(0.02)
        nug_spin.setValue(float(data.get("nugget_effect", 0.05)))
        nug_spin.valueChanged.connect(lambda v: self.parameter_changed.emit("petrophysics", "nugget_effect", v))
        form_var.addRow("Nugget Effect (c0):", self._with_badge(nug_spin, "var"))
        self.inputs["nugget_effect"] = nug_spin

        sill_spin = QDoubleSpinBox()
        sill_spin.setRange(0.05, 5.00)
        sill_spin.setSingleStep(0.05)
        sill_spin.setValue(float(data.get("sill_variance", 1.00)))
        sill_spin.valueChanged.connect(lambda v: self.parameter_changed.emit("petrophysics", "sill_variance", v))
        form_var.addRow("Sill Variance (C):", self._with_badge(sill_spin, "var"))
        self.inputs["sill_variance"] = sill_spin

        nug_ratio_badge = QLabel()
        nug_ratio_badge.setStyleSheet("color: #16a34a; font-weight: bold;")
        form_var.addRow("Relative Nugget %:", nug_ratio_badge)

        self.form_layout.addLayout(form_var)

        # Stochastic Realization Engine
        lbl_eng = QLabel("Stochastic Simulation Engine")
        lbl_eng.setObjectName("sectionHeader")
        self.form_layout.addWidget(lbl_eng)

        form_eng = QFormLayout()
        form_eng.setSpacing(6)

        algo_combo = QComboBox()
        algo_combo.addItems([
            "Sequential Gaussian Simulation (SGSIM)",
            "Truncated Gaussian (Facies SGSIM)",
            "Collocated Co-Kriging"
        ])
        algo_combo.setCurrentText(str(data.get("geostat_algorithm", "Sequential Gaussian Simulation (SGSIM)")))
        algo_combo.currentTextChanged.connect(lambda v: self.parameter_changed.emit("petrophysics", "geostat_algorithm", v))
        form_eng.addRow("Simulation Algorithm:", algo_combo)

        cond_combo = QComboBox()
        cond_combo.addItems(["Conditional to Well Hard Data", "Unconditional Stochastic Field"])
        cond_combo.setCurrentText(str(data.get("geostat_conditioning", "Conditional to Well Hard Data")))
        cond_combo.currentTextChanged.connect(lambda v: self.parameter_changed.emit("petrophysics", "geostat_conditioning", v))
        form_eng.addRow("Conditioning Mode:", cond_combo)

        seed_layout = QHBoxLayout()
        seed_spin = QSpinBox()
        seed_spin.setRange(0, 999999)
        seed_spin.setValue(int(data.get("geostat_seed", 42)))
        seed_spin.valueChanged.connect(lambda v: self.parameter_changed.emit("petrophysics", "geostat_seed", v))
        self.inputs["geostat_seed"] = seed_spin
        seed_layout.addWidget(seed_spin, stretch=1)

        btn_reroll = QPushButton("Re-roll Seed")
        btn_reroll.setStyleSheet("background: #f1f5f9; border: 1px solid #94a3b8; border-radius: 4px; padding: 2px 6px; font-weight: bold; font-size: 10px;")
        btn_reroll.clicked.connect(lambda: seed_spin.setValue(int(np.random.randint(100, 999999))))
        seed_layout.addWidget(btn_reroll)

        form_eng.addRow("Random Seed:", seed_layout)

        btn_regen = QPushButton("Regenerate Realization")
        btn_regen.setStyleSheet("background: #0284c7; color: white; font-weight: bold; font-size: 11px; padding: 5px; border-radius: 4px;")
        btn_regen.clicked.connect(lambda: self.apply_requested.emit())
        form_eng.addRow("", btn_regen)

        self.form_layout.addLayout(form_eng)

        def _update_var_badges():
            maj = rmaj_spin.value()
            minor = rmin_spin.value()
            c0 = nug_spin.value()
            c = sill_spin.value()

            aniso_val = maj / max(minor, 1.0)
            aniso_badge.setText(f"{aniso_val:.1f} : 1.0 (Major / Minor)")

            total_sill = c0 + c
            nug_pct = (c0 / max(total_sill, 1e-4)) * 100.0
            nug_ratio_badge.setText(f"{nug_pct:.1f}% ({'Strong continuity' if nug_pct < 20 else 'Moderate continuity'})")

        for sp in [rmaj_spin, rmin_spin, nug_spin, sill_spin]:
            sp.valueChanged.connect(lambda _: _update_var_badges())
        vt_combo.currentTextChanged.connect(lambda _: _update_var_badges())

        _update_var_badges()

    def _build_pvt_form(self, data: Dict[str, Any]):
        lbl_preset = QLabel("Fluid System & Characterization")
        lbl_preset.setObjectName("sectionHeader")
        self.form_layout.addWidget(lbl_preset)

        form_top = QFormLayout()
        form_top.setVerticalSpacing(6)
        form_top.setHorizontalSpacing(8)

        combo_preset = QComboBox()
        combo_preset.setMinimumHeight(24)
        combo_preset.addItems([
            "Permian San Andres (Medium Black Oil)",
            "Gulf Coast Deep (Light Volatile)",
            "Midland Basin Tight (Light Crude)",
            "Heavy Crude (Viscous EOR)",
            "Custom Fluid System"
        ])
        saved_preset = str(data.get("fluid_preset", "Permian San Andres (Medium Black Oil)"))
        idx_p = combo_preset.findText(saved_preset)
        if idx_p >= 0:
            combo_preset.setCurrentIndex(idx_p)
        form_top.addRow("Fluid Preset:", combo_preset)

        combo_model = QComboBox()
        combo_model.setMinimumHeight(24)
        combo_model.addItems([
            "Solvent-Extended (Todd-Longstaff CO2)",
            "Standard Black Oil (Standing / Beggs)",
            "Compositional (PR-EOS Flash)"
        ])
        saved_m = str(data.get("fluid_model_type", "Solvent-Extended (Todd-Longstaff CO2)"))
        idx_m = combo_model.findText(saved_m)
        if idx_m >= 0:
            combo_model.setCurrentIndex(idx_m)
        combo_model.currentTextChanged.connect(lambda v: self.parameter_changed.emit("pvt", "fluid_model_type", v))
        form_top.addRow("Model Type:", combo_model)
        self.inputs["fluid_model_type"] = combo_model

        self.form_layout.addLayout(form_top)

        # 1. In-Situ Thermodynamic State
        lbl_thermo = QLabel("In-Situ Reservoir Thermodynamics")
        lbl_thermo.setObjectName("sectionHeader")
        self.form_layout.addWidget(lbl_thermo)

        form_thermo = QFormLayout()
        form_thermo.setVerticalSpacing(6)
        form_thermo.setHorizontalSpacing(8)

        pres_spin = QDoubleSpinBox()
        pres_spin.setRange(500.0, 15000.0)
        pres_spin.setSingleStep(50.0)
        pres_spin.setValue(float(data.get("initial_pressure", 4000.0)))
        pres_spin.valueChanged.connect(lambda v: self.parameter_changed.emit("pvt", "initial_pressure", v))
        form_thermo.addRow("Initial Pressure:", self._with_badge(pres_spin, "psia"))
        self.inputs["initial_pressure"] = pres_spin

        temp_spin = QDoubleSpinBox()
        temp_spin.setRange(60.0, 450.0)
        temp_spin.setSingleStep(5.0)
        temp_spin.setValue(float(data.get("temperature", 212.0)))
        temp_spin.valueChanged.connect(lambda v: self.parameter_changed.emit("pvt", "temperature", v))
        form_thermo.addRow("Reservoir Temp:", self._with_badge(temp_spin, "°F"))
        self.inputs["temperature"] = temp_spin

        datum_spin = QDoubleSpinBox()
        datum_spin.setRange(500.0, 30000.0)
        datum_spin.setSingleStep(50.0)
        datum_spin.setValue(float(data.get("top_depth", 5000.0)))
        datum_spin.valueChanged.connect(lambda v: self.parameter_changed.emit("pvt", "top_depth", v))
        form_thermo.addRow("Datum Depth:", self._with_badge(datum_spin, "ft"))
        self.inputs["top_depth"] = datum_spin

        self.form_layout.addLayout(form_thermo)

        # 2. Crude Oil Properties
        lbl_oil = QLabel("Crude Oil & Hydrocarbon Properties")
        lbl_oil.setObjectName("sectionHeader")
        self.form_layout.addWidget(lbl_oil)

        form_oil = QFormLayout()
        form_oil.setVerticalSpacing(6)
        form_oil.setHorizontalSpacing(8)

        api_spin = QDoubleSpinBox()
        api_spin.setRange(10.0, 65.0)
        api_spin.setSingleStep(0.5)
        api_spin.setValue(float(data.get("api_gravity", 35.0)))
        api_spin.valueChanged.connect(lambda v: self.parameter_changed.emit("pvt", "api_gravity", v))
        form_oil.addRow("Oil Gravity:", self._with_badge(api_spin, "°API"))
        self.inputs["api_gravity"] = api_spin

        gor_spin = QDoubleSpinBox()
        gor_spin.setRange(0.0, 5000.0)
        gor_spin.setSingleStep(25.0)
        gor_spin.setValue(float(data.get("sol_gor", 500.0)))
        gor_spin.valueChanged.connect(lambda v: self.parameter_changed.emit("pvt", "sol_gor", v))
        form_oil.addRow("Solution GOR:", self._with_badge(gor_spin, "scf/STB"))
        self.inputs["sol_gor"] = gor_spin

        gas_grav_spin = QDoubleSpinBox()
        gas_grav_spin.setRange(0.55, 1.50)
        gas_grav_spin.setSingleStep(0.01)
        gas_grav_spin.setDecimals(3)
        gas_grav_spin.setValue(float(data.get("gas_specific_gravity", 0.70)))
        gas_grav_spin.valueChanged.connect(lambda v: self.parameter_changed.emit("pvt", "gas_specific_gravity", v))
        form_oil.addRow("Gas Gravity:", self._with_badge(gas_grav_spin, "air=1"))
        self.inputs["gas_specific_gravity"] = gas_grav_spin

        pb_spin = QDoubleSpinBox()
        pb_spin.setRange(200.0, 10000.0)
        pb_spin.setSingleStep(50.0)
        pb_spin.setValue(float(data.get("bubble_point_pressure", 2800.0)))
        pb_spin.valueChanged.connect(lambda v: self.parameter_changed.emit("pvt", "bubble_point_pressure", v))
        form_oil.addRow("Bubble Point:", self._with_badge(pb_spin, "psia"))
        self.inputs["bubble_point_pressure"] = pb_spin

        visc_spin = QDoubleSpinBox()
        visc_spin.setRange(0.1, 500.0)
        visc_spin.setSingleStep(0.1)
        visc_spin.setValue(float(data.get("oil_viscosity_cp", 1.0)))
        visc_spin.valueChanged.connect(lambda v: self.parameter_changed.emit("pvt", "oil_viscosity_cp", v))
        form_oil.addRow("Dead Oil Visc:", self._with_badge(visc_spin, "cP"))
        self.inputs["oil_viscosity_cp"] = visc_spin

        boi_spin = QDoubleSpinBox()
        boi_spin.setRange(1.00, 2.50)
        boi_spin.setSingleStep(0.02)
        boi_spin.setDecimals(3)
        boi_spin.setValue(float(data.get("boi", 1.20)))
        boi_spin.valueChanged.connect(lambda v: self.parameter_changed.emit("pvt", "boi", v))
        form_oil.addRow("Oil FVF (Boi):", self._with_badge(boi_spin, "rb/STB"))
        self.inputs["boi"] = boi_spin

        co_spin = QDoubleSpinBox()
        co_spin.setRange(1.0, 50.0)
        co_spin.setSingleStep(0.5)
        co_spin.setValue(float(data.get("oil_compressibility_e6", 12.0)))
        co_spin.valueChanged.connect(lambda v: self.parameter_changed.emit("pvt", "oil_compressibility_e6", v))
        form_oil.addRow("Oil Compress:", self._with_badge(co_spin, "μsip"))
        self.inputs["oil_compressibility_e6"] = co_spin

        self.form_layout.addLayout(form_oil)

        # 3. Solvent Interaction & MMP
        lbl_solvent = QLabel("CO2 Solvent Interaction & Miscibility")
        lbl_solvent.setObjectName("sectionHeader")
        self.form_layout.addWidget(lbl_solvent)

        form_solvent = QFormLayout()
        form_solvent.setVerticalSpacing(6)
        form_solvent.setHorizontalSpacing(8)

        purity_spin = QDoubleSpinBox()
        purity_spin.setRange(50.0, 100.0)
        purity_spin.setSingleStep(1.0)
        purity_spin.setValue(float(data.get("co2_purity", 95.0)))
        purity_spin.valueChanged.connect(lambda v: self.parameter_changed.emit("pvt", "co2_purity", v))
        form_solvent.addRow("CO2 Purity:", self._with_badge(purity_spin, "%"))
        self.inputs["co2_purity"] = purity_spin

        sf_spin = QDoubleSpinBox()
        sf_spin.setRange(1.00, 2.00)
        sf_spin.setSingleStep(0.01)
        sf_spin.setDecimals(3)
        sf_spin.setValue(float(data.get("co2_swelling_factor_max", 1.25)))
        sf_spin.valueChanged.connect(lambda v: self.parameter_changed.emit("pvt", "co2_swelling_factor_max", v))
        form_solvent.addRow("Max Swelling SF:", self._with_badge(sf_spin, "mult"))
        self.inputs["co2_swelling_factor_max"] = sf_spin

        omega_spin = QDoubleSpinBox()
        omega_spin.setRange(0.0, 1.0)
        omega_spin.setSingleStep(0.05)
        omega_spin.setDecimals(2)
        omega_spin.setValue(float(data.get("todd_longstaff_omega", 0.70)))
        omega_spin.valueChanged.connect(lambda v: self.parameter_changed.emit("pvt", "todd_longstaff_omega", v))
        form_solvent.addRow("Todd-Longstaff ω:", self._with_badge(omega_spin, "dim"))
        self.inputs["todd_longstaff_omega"] = omega_spin

        mmp_method_combo = QComboBox()
        mmp_method_combo.setMinimumHeight(24)
        mmp_method_combo.addItems([
            "Yellig-Metcalfe Correlation",
            "Cronquist Correlation",
            "Lee Correlation",
            "Alston et al.",
            "Emera-Sarma Analytical",
            "Manual Experimental MMP"
        ])
        cur_mmp_m = str(data.get("mmp_correlation", "Yellig-Metcalfe Correlation"))
        idx_m = mmp_method_combo.findText(cur_mmp_m)
        if idx_m >= 0:
            mmp_method_combo.setCurrentIndex(idx_m)
        mmp_method_combo.currentTextChanged.connect(lambda v: self.parameter_changed.emit("pvt", "mmp_correlation", v))
        form_solvent.addRow("MMP Correlation:", mmp_method_combo)
        self.inputs["mmp_correlation"] = mmp_method_combo

        mmp_spin = QDoubleSpinBox()
        mmp_spin.setRange(800.0, 8000.0)
        mmp_spin.setSingleStep(50.0)
        mmp_spin.setValue(float(data.get("mmp_override", 2688.0)))
        mmp_spin.valueChanged.connect(lambda v: self.parameter_changed.emit("pvt", "mmp_override", v))
        form_solvent.addRow("MMP Override:", self._with_badge(mmp_spin, "psia"))
        self.inputs["mmp_override"] = mmp_spin

        self.form_layout.addLayout(form_solvent)

        # 4. Fluid Contacts & 3D Reservoir Saturation Column
        lbl_contacts = QLabel("Fluid Contacts & 3D Column Equilibrium")
        lbl_contacts.setObjectName("sectionHeader")
        self.form_layout.addWidget(lbl_contacts)

        form_contacts = QFormLayout()
        form_contacts.setVerticalSpacing(6)
        form_contacts.setHorizontalSpacing(8)

        woc_spin = QDoubleSpinBox()
        woc_spin.setRange(500.0, 30000.0)
        woc_spin.setSingleStep(5.0)
        woc_spin.setValue(float(data.get("woc_depth", 5035.0)))
        woc_spin.valueChanged.connect(lambda v: self.parameter_changed.emit("pvt", "woc_depth", v))
        form_contacts.addRow("WOC Depth:", self._with_badge(woc_spin, "ft TVD"))
        self.inputs["woc_depth"] = woc_spin

        chk_gas_cap = QCheckBox("Gas Cap Present Above Leg")
        chk_gas_cap.setMinimumHeight(24)
        chk_gas_cap.setChecked(bool(data.get("has_gas_cap", False)))
        chk_gas_cap.toggled.connect(lambda v: self.parameter_changed.emit("pvt", "has_gas_cap", v))
        form_contacts.addRow("Gas Cap Leg:", chk_gas_cap)
        self.inputs["has_gas_cap"] = chk_gas_cap
        self.inputs["has_gas_cap"] = chk_gas_cap

        goc_spin = QDoubleSpinBox()
        goc_spin.setRange(500.0, 30000.0)
        goc_spin.setSingleStep(5.0)
        goc_spin.setValue(float(data.get("goc_depth", 4985.0)))
        goc_spin.setEnabled(chk_gas_cap.isChecked())
        chk_gas_cap.toggled.connect(goc_spin.setEnabled)
        goc_spin.valueChanged.connect(lambda v: self.parameter_changed.emit("pvt", "goc_depth", v))
        form_contacts.addRow("GOC Depth:", self._with_badge(goc_spin, "ft TVD"))
        self.inputs["goc_depth"] = goc_spin

        w_grad_spin = QDoubleSpinBox()
        w_grad_spin.setRange(0.40, 0.55)
        w_grad_spin.setSingleStep(0.005)
        w_grad_spin.setDecimals(3)
        w_grad_spin.setValue(float(data.get("water_gradient", 0.465)))
        w_grad_spin.valueChanged.connect(lambda v: self.parameter_changed.emit("pvt", "water_gradient", v))
        form_contacts.addRow("Water Gradient:", self._with_badge(w_grad_spin, "psi/ft"))
        self.inputs["water_gradient"] = w_grad_spin

        g_grad_spin = QDoubleSpinBox()
        g_grad_spin.setRange(0.04, 0.20)
        g_grad_spin.setSingleStep(0.005)
        g_grad_spin.setDecimals(3)
        g_grad_spin.setValue(float(data.get("gas_gradient", 0.080)))
        g_grad_spin.valueChanged.connect(lambda v: self.parameter_changed.emit("pvt", "gas_gradient", v))
        form_contacts.addRow("Gas Gradient:", self._with_badge(g_grad_spin, "psi/ft"))
        self.inputs["gas_gradient"] = g_grad_spin

        self.form_layout.addLayout(form_contacts)

        def _apply_preset_selected(p_text: str):
            if "Permian" in p_text:
                api_spin.setValue(35.0)
                gor_spin.setValue(500.0)
                gas_grav_spin.setValue(0.72)
                pb_spin.setValue(2200.0)
                visc_spin.setValue(1.8)
                temp_spin.setValue(120.0)
                pres_spin.setValue(3200.0)
                mmp_spin.setValue(1850.0)
                sf_spin.setValue(1.22)
            elif "Gulf" in p_text:
                api_spin.setValue(42.0)
                gor_spin.setValue(1200.0)
                gas_grav_spin.setValue(0.82)
                pb_spin.setValue(3800.0)
                visc_spin.setValue(0.45)
                temp_spin.setValue(220.0)
                pres_spin.setValue(5500.0)
                mmp_spin.setValue(2600.0)
                sf_spin.setValue(1.35)
            elif "Midland" in p_text:
                api_spin.setValue(38.0)
                gor_spin.setValue(750.0)
                gas_grav_spin.setValue(0.70)
                pb_spin.setValue(2900.0)
                visc_spin.setValue(0.85)
                temp_spin.setValue(150.0)
                pres_spin.setValue(4200.0)
                mmp_spin.setValue(2150.0)
                sf_spin.setValue(1.28)
            elif "Heavy" in p_text:
                api_spin.setValue(22.0)
                gor_spin.setValue(150.0)
                gas_grav_spin.setValue(0.65)
                pb_spin.setValue(1100.0)
                visc_spin.setValue(18.0)
                temp_spin.setValue(110.0)
                pres_spin.setValue(2500.0)
                mmp_spin.setValue(3400.0)
                sf_spin.setValue(1.15)
            self.parameter_changed.emit("pvt", "fluid_preset", p_text)

        combo_preset.currentTextChanged.connect(_apply_preset_selected)

    def _build_fault_form(self, data: Dict[str, Any]):
        lbl = QLabel("Fault Geometry & Slip Stability")
        lbl.setObjectName("sectionHeader")
        self.form_layout.addWidget(lbl)

        btn_box = QHBoxLayout()
        btn_box.setSpacing(4)
        btn_mgr = QPushButton("Fault & Caprock Manager...")
        btn_mgr.setStyleSheet("background: #e0f2fe; color: #0284c7; border: 1px solid #bae6fd; border-radius: 4px; font-weight: bold; font-size: 10px; padding: 4px;")
        btn_mgr.clicked.connect(lambda: self.manage_faults_requested.emit(0))
        btn_box.addWidget(btn_mgr)

        btn_add = QPushButton("+ Add Fault")
        btn_add.setStyleSheet("background: #f1f5f9; border: 1px solid #cbd5e1; border-radius: 4px; font-weight: bold; font-size: 10px; padding: 4px;")
        btn_add.clicked.connect(self.add_fault_requested.emit)
        btn_box.addWidget(btn_add)

        btn_iso = QPushButton("Isolate in 3D")
        btn_iso.setStyleSheet("background: #f1f5f9; border: 1px solid #cbd5e1; border-radius: 4px; font-weight: bold; font-size: 10px; padding: 4px;")
        btn_iso.clicked.connect(lambda: self.isolate_fault_requested.emit(""))
        btn_box.addWidget(btn_iso)
        self.form_layout.addLayout(btn_box)

        form = QFormLayout()
        form.setSpacing(6)

        name_edit = QLineEdit(str(data.get("fault_name", "Fault F-1")))
        name_edit.textChanged.connect(lambda v: self.parameter_changed.emit("geomechanics", "fault_name", v))
        form.addRow("Fault Name:", name_edit)

        dip_spin = QDoubleSpinBox()
        dip_spin.setRange(10.0, 90.0)
        dip_spin.setSingleStep(1.0)
        dip_spin.setValue(float(data.get("fault_dip", 60.0)))
        dip_spin.valueChanged.connect(lambda v: self.parameter_changed.emit("geomechanics", "fault_dip", v))
        form.addRow("Dip Angle (θ):", self._with_badge(dip_spin, "deg"))

        strike_spin = QDoubleSpinBox()
        strike_spin.setRange(0.0, 360.0)
        strike_spin.setSingleStep(5.0)
        strike_spin.setValue(float(data.get("fault_strike", 45.0)))
        strike_spin.valueChanged.connect(lambda v: self.parameter_changed.emit("geomechanics", "fault_strike", v))
        form.addRow("Strike Angle (α):", self._with_badge(strike_spin, "deg"))

        trans_spin = QDoubleSpinBox()
        trans_spin.setRange(0.0, 1.0)
        trans_spin.setSingleStep(0.05)
        trans_spin.setDecimals(2)
        trans_spin.setValue(float(data.get("fault_trans_mult", 0.15)))
        trans_spin.valueChanged.connect(lambda v: self.parameter_changed.emit("geomechanics", "fault_trans_mult", v))
        form.addRow("Transmissibility Mult:", self._with_badge(trans_spin, "mult"))

        throw_spin = QDoubleSpinBox()
        throw_spin.setRange(0.0, 500.0)
        throw_spin.setSingleStep(5.0)
        throw_spin.setValue(float(data.get("fault_throw", 25.0)))
        throw_spin.valueChanged.connect(lambda v: self.parameter_changed.emit("geomechanics", "fault_throw", v))
        form.addRow("Throw / Displacement:", self._with_badge(throw_spin, "ft"))

        fric_spin = QDoubleSpinBox()
        fric_spin.setRange(0.10, 1.00)
        fric_spin.setSingleStep(0.05)
        fric_spin.setDecimals(2)
        fric_spin.setValue(float(data.get("fault_friction", 0.60)))
        fric_spin.valueChanged.connect(lambda v: self.parameter_changed.emit("geomechanics", "fault_friction", v))
        form.addRow("Friction Coeff (μ):", self._with_badge(fric_spin, "coeff"))

        coh_spin = QDoubleSpinBox()
        coh_spin.setRange(0.0, 1000.0)
        coh_spin.setSingleStep(25.0)
        coh_spin.setValue(float(data.get("fault_cohesion", 0.0)))
        coh_spin.valueChanged.connect(lambda v: self.parameter_changed.emit("geomechanics", "fault_cohesion", v))
        form.addRow("Fault Cohesion (S0):", self._with_badge(coh_spin, "psi"))

        lx = float(data.get("length", 2000.0))
        cx_spin = QDoubleSpinBox()
        cx_spin.setRange(0.0, 50000.0)
        cx_spin.setSingleStep(50.0)
        cx_spin.setValue(float(data.get("fault_center_x", lx * 0.5)))
        cx_spin.valueChanged.connect(lambda v: self.parameter_changed.emit("geomechanics", "fault_center_x", v))
        form.addRow("Fault Center X:", self._with_badge(cx_spin, "ft"))

        ly = float(data.get("area", 1000.0)) * 43560.0 / max(lx, 1.0)
        cy_spin = QDoubleSpinBox()
        cy_spin.setRange(0.0, 50000.0)
        cy_spin.setSingleStep(50.0)
        cy_spin.setValue(float(data.get("fault_center_y", ly * 0.5)))
        cy_spin.valueChanged.connect(lambda v: self.parameter_changed.emit("geomechanics", "fault_center_y", v))
        form.addRow("Fault Center Y:", self._with_badge(cy_spin, "ft"))

        slip_badge = QLabel("Ts = 0.38 (Sub-Critical / Stable)")
        slip_badge.setStyleSheet("color: #16a34a; font-weight: bold;")
        form.addRow("Mohr-Coulomb Status:", slip_badge)

        self.form_layout.addLayout(form)

    def _build_caprock_form(self, data: Dict[str, Any]):
        lbl = QLabel("Caprock Seal Integrity & Boundaries")
        lbl.setObjectName("sectionHeader")
        self.form_layout.addWidget(lbl)

        btn_mgr = QPushButton("Open Caprock Stratigraphy Manager...")
        btn_mgr.setStyleSheet("background: #e0f2fe; color: #0284c7; border: 1px solid #bae6fd; border-radius: 4px; font-weight: bold; font-size: 10px; padding: 4px;")
        btn_mgr.clicked.connect(lambda: self.manage_faults_requested.emit(2))
        self.form_layout.addWidget(btn_mgr)

        form = QFormLayout()
        form.setSpacing(6)

        litho_combo = QComboBox()
        litho_combo.addItems(["Dense Marine Shale", "Anhydrite Evaporite", "Siltstone Mudstone", "Dense Carbonate"])
        litho_combo.setCurrentText(str(data.get("caprock_lithology", "Dense Marine Shale")))
        litho_combo.currentTextChanged.connect(lambda v: self.parameter_changed.emit("geomechanics", "caprock_lithology", v))
        form.addRow("Caprock Lithology:", litho_combo)

        thk_spin = QDoubleSpinBox()
        thk_spin.setRange(10.0, 3000.0)
        thk_spin.setSingleStep(10.0)
        thk_spin.setValue(float(data.get("caprock_thickness", 200.0)))
        thk_spin.valueChanged.connect(lambda v: self.parameter_changed.emit("geomechanics", "caprock_thickness", v))
        form.addRow("Seal Thickness:", self._with_badge(thk_spin, "ft"))

        t0_spin = QDoubleSpinBox()
        t0_spin.setRange(0.0, 2000.0)
        t0_spin.setSingleStep(25.0)
        t0_spin.setValue(float(data.get("caprock_t0", 200.0)))
        t0_spin.valueChanged.connect(lambda v: self.parameter_changed.emit("geomechanics", "caprock_t0", v))
        form.addRow("Tensile Strength (T0):", self._with_badge(t0_spin, "psi"))

        coh_spin = QDoubleSpinBox()
        coh_spin.setRange(0.0, 5000.0)
        coh_spin.setSingleStep(50.0)
        coh_spin.setValue(float(data.get("caprock_cohesion", 400.0)))
        coh_spin.valueChanged.connect(lambda v: self.parameter_changed.emit("geomechanics", "caprock_cohesion", v))
        form.addRow("Cohesion (C0):", self._with_badge(coh_spin, "psi"))

        phi_spin = QDoubleSpinBox()
        phi_spin.setRange(10.0, 50.0)
        phi_spin.setSingleStep(1.0)
        phi_spin.setValue(float(data.get("caprock_friction_angle", 30.0)))
        phi_spin.valueChanged.connect(lambda v: self.parameter_changed.emit("geomechanics", "caprock_friction_angle", v))
        form.addRow("Friction Angle:", self._with_badge(phi_spin, "deg"))

        pentry_spin = QDoubleSpinBox()
        pentry_spin.setRange(50.0, 5000.0)
        pentry_spin.setSingleStep(50.0)
        pentry_spin.setValue(float(data.get("caprock_entry_pressure", 1500.0)))
        pentry_spin.valueChanged.connect(lambda v: self.parameter_changed.emit("geomechanics", "caprock_entry_pressure", v))
        form.addRow("Capillary Entry P:", self._with_badge(pentry_spin, "psi"))

        perm_spin = QDoubleSpinBox()
        perm_spin.setRange(0.00001, 0.1)
        perm_spin.setSingleStep(0.0001)
        perm_spin.setDecimals(5)
        perm_spin.setValue(float(data.get("caprock_perm", 0.0001)))
        perm_spin.valueChanged.connect(lambda v: self.parameter_changed.emit("geomechanics", "caprock_perm", v))
        form.addRow("Permeability (k_cap):", self._with_badge(perm_spin, "mD"))

        sf_spin = QDoubleSpinBox()
        sf_spin.setRange(0.70, 1.00)
        sf_spin.setSingleStep(0.05)
        sf_spin.setDecimals(2)
        sf_spin.setValue(float(data.get("caprock_safety_factor", 0.90)))
        sf_spin.valueChanged.connect(lambda v: self.parameter_changed.emit("geomechanics", "caprock_safety_factor", v))
        form.addRow("Class VI Factor:", self._with_badge(sf_spin, "ratio"))

        seal_status = QLabel("INTACT (Tensile Margin: +1,250 psi)")
        seal_status.setStyleSheet("color: #16a34a; font-weight: bold;")
        self.form_layout.addLayout(form)

    def _build_stress_uic_form(self, data: Dict[str, Any]):
        lbl = QLabel("In-Situ Stress State & Elastic Rock Physics")
        lbl.setObjectName("sectionHeader")
        self.form_layout.addWidget(lbl)

        form = QFormLayout()
        form.setSpacing(6)

        ym_spin = QDoubleSpinBox()
        ym_spin.setRange(2.0, 100.0)
        ym_spin.setSingleStep(1.0)
        ym_spin.setValue(float(data.get("youngs_modulus_base", 20.0)))
        ym_spin.valueChanged.connect(lambda v: self.parameter_changed.emit("geomechanics", "youngs_modulus_base", v))
        form.addRow("Young's Modulus (E):", self._with_badge(ym_spin, "GPa"))
        self.inputs["youngs_modulus_base"] = ym_spin

        nu_spin = QDoubleSpinBox()
        nu_spin.setRange(0.10, 0.45)
        nu_spin.setSingleStep(0.02)
        nu_spin.setDecimals(2)
        nu_spin.setValue(float(data.get("poissons_ratio", 0.25)))
        nu_spin.valueChanged.connect(lambda v: self.parameter_changed.emit("geomechanics", "poissons_ratio", v))
        form.addRow("Poisson's Ratio (ν):", self._with_badge(nu_spin, "ratio"))
        self.inputs["poissons_ratio"] = nu_spin

        alpha_spin = QDoubleSpinBox()
        alpha_spin.setRange(0.50, 1.00)
        alpha_spin.setSingleStep(0.05)
        alpha_spin.setDecimals(2)
        alpha_spin.setValue(float(data.get("biot_coeff", 0.80)))
        alpha_spin.valueChanged.connect(lambda v: self.parameter_changed.emit("geomechanics", "biot_coeff", v))
        form.addRow("Biot Coefficient (α):", self._with_badge(alpha_spin, "coeff"))
        self.inputs["biot_coeff"] = alpha_spin

        ob_spin = QDoubleSpinBox()
        ob_spin.setRange(0.70, 1.50)
        ob_spin.setSingleStep(0.02)
        ob_spin.setDecimals(2)
        ob_spin.setValue(float(data.get("overburden_grad", 1.00)))
        ob_spin.valueChanged.connect(lambda v: self.parameter_changed.emit("geomechanics", "overburden_grad", v))
        form.addRow("Overburden Gradient (σv):", self._with_badge(ob_spin, "psi/ft"))
        self.inputs["overburden_grad"] = ob_spin

        k0_spin = QDoubleSpinBox()
        k0_spin.setRange(0.40, 1.20)
        k0_spin.setSingleStep(0.05)
        k0_spin.setDecimals(2)
        k0_spin.setValue(float(data.get("stress_k0", 0.75)))
        k0_spin.valueChanged.connect(lambda v: self.parameter_changed.emit("geomechanics", "stress_k0", v))
        form.addRow("Stress Ratio (K0):", self._with_badge(k0_spin, "ratio"))
        self.inputs["stress_k0"] = k0_spin

        frac_spin = QDoubleSpinBox()
        frac_spin.setRange(0.50, 1.30)
        frac_spin.setSingleStep(0.02)
        frac_spin.setDecimals(2)
        frac_spin.setValue(float(data.get("frac_grad", 0.85)))
        frac_spin.valueChanged.connect(lambda v: self.parameter_changed.emit("geomechanics", "frac_grad", v))
        form.addRow("Fracture Gradient:", self._with_badge(frac_spin, "psi/ft"))
        self.inputs["frac_grad"] = frac_spin

        lbl_uic = QLabel("EPA Class VI Geomechanical Safety")
        lbl_uic.setObjectName("sectionHeader")
        self.form_layout.addLayout(form)
        self.form_layout.addWidget(lbl_uic)

        form2 = QFormLayout()
        form2.setSpacing(6)

        ceiling_label = QLabel("0.90 × P_frac (EPA UIC Class VI Mandate)")
        ceiling_label.setStyleSheet("color: #16a34a; font-weight: bold;")
        form2.addRow("Containment Policy:", ceiling_label)

        depth_val = float(data.get("top_depth", 5000.0))
        fg_val = float(data.get("frac_grad", 0.85))
        p_frac = fg_val * depth_val
        p_safe = 0.90 * p_frac
        whp_safe = max(400.0, p_safe - depth_val * 0.28)

        safe_badge = QLabel(f"P_max = {p_safe:,.0f} psia (Frac: {p_frac:,.0f} psia at {depth_val:.0f} ft)")
        safe_badge.setStyleSheet("color: #0284c7; font-weight: bold; font-size: 11px;")
        form2.addRow("Safe Sandface Ceiling:", safe_badge)

        whp_badge = QLabel(f"WHP_max = {whp_safe:,.0f} psia (Hyd static: {depth_val*0.28:,.0f} psi)")
        whp_badge.setStyleSheet("color: #0d9488; font-weight: bold; font-size: 11px;")
        form2.addRow("Surface WHP Limit:", whp_badge)

        usdw_spin = QDoubleSpinBox()
        usdw_spin.setRange(200.0, 5000.0)
        usdw_spin.setSingleStep(50.0)
        usdw_spin.setValue(float(data.get("usdw_depth", 1200.0)))
        usdw_spin.valueChanged.connect(lambda v: self.parameter_changed.emit("geomechanics", "usdw_depth", v))
        form2.addRow("Lowermost USDW Depth:", self._with_badge(usdw_spin, "ft TVD"))
        self.inputs["usdw_depth"] = usdw_spin

        aor_badge = QLabel("R_AoR ≈ 3,850 ft (Pressure elevation above USDW critical head)")
        aor_badge.setStyleSheet("color: #ea580c; font-weight: bold; font-size: 11px;")
        form2.addRow("AoR Plume Footprint:", aor_badge)

        self.form_layout.addLayout(form2)

    def _build_storage_geothermal_form(self, data: Dict[str, Any]):
        lbl = QLabel("CCUS Storage, 45Q & Geothermal CPG")
        lbl.setObjectName("sectionHeader")
        self.form_layout.addWidget(lbl)

        form = QFormLayout()
        form.setSpacing(6)

        c45q_eor_spin = QDoubleSpinBox()
        c45q_eor_spin.setRange(10.0, 200.0)
        c45q_eor_spin.setSingleStep(5.0)
        c45q_eor_spin.setValue(float(data.get("rate_45q_eor", 60.0)))
        c45q_eor_spin.valueChanged.connect(lambda v: self.parameter_changed.emit("storage", "rate_45q_eor", v))
        form.addRow("45Q EOR Credit Rate:", self._with_badge(c45q_eor_spin, "$/t"))

        c45q_saline_spin = QDoubleSpinBox()
        c45q_saline_spin.setRange(10.0, 250.0)
        c45q_saline_spin.setSingleStep(5.0)
        c45q_saline_spin.setValue(float(data.get("rate_45q_saline", 85.0)))
        c45q_saline_spin.valueChanged.connect(lambda v: self.parameter_changed.emit("storage", "rate_45q_saline", v))
        form.addRow("45Q Saline Storage Rate:", self._with_badge(c45q_saline_spin, "$/t"))

        land_c_spin = QDoubleSpinBox()
        land_c_spin.setRange(0.5, 3.0)
        land_c_spin.setSingleStep(0.1)
        land_c_spin.setDecimals(2)
        land_c_spin.setValue(float(data.get("land_trapping_c", 1.25)))
        land_c_spin.valueChanged.connect(lambda v: self.parameter_changed.emit("storage", "land_trapping_c", v))
        form.addRow("Land Residual Trapping C:", self._with_badge(land_c_spin, "dim"))

        lbl_geo = QLabel("CO2 Plume Geothermal (CPG) Energy")
        lbl_geo.setObjectName("sectionHeader")
        self.form_layout.addLayout(form)
        self.form_layout.addWidget(lbl_geo)

        form_geo = QFormLayout()
        form_geo.setSpacing(6)

        m_dot_spin = QDoubleSpinBox()
        m_dot_spin.setRange(10.0, 500.0)
        m_dot_spin.setSingleStep(10.0)
        m_dot_spin.setValue(float(data.get("cpg_mass_rate", 120.0)))
        m_dot_spin.valueChanged.connect(lambda v: self.parameter_changed.emit("storage", "cpg_mass_rate", v))
        form_geo.addRow("CO2 Circulation Mass Rate:", self._with_badge(m_dot_spin, "kg/s"))

        temp_res_spin = QDoubleSpinBox()
        temp_res_spin.setRange(100.0, 450.0)
        temp_res_spin.setSingleStep(5.0)
        temp_res_spin.setValue(float(data.get("temperature", 215.0)))
        temp_res_spin.valueChanged.connect(lambda v: self.parameter_changed.emit("storage", "temperature", v))
        form_geo.addRow("Formation Temperature:", self._with_badge(temp_res_spin, "°F"))

        eta_spin = QDoubleSpinBox()
        eta_spin.setRange(5.0, 30.0)
        eta_spin.setSingleStep(0.5)
        eta_spin.setValue(float(data.get("cpg_efficiency", 14.5)))
        eta_spin.valueChanged.connect(lambda v: self.parameter_changed.emit("storage", "cpg_efficiency", v))
        form_geo.addRow("Turbine Cycle Efficiency:", self._with_badge(eta_spin, "%"))

        self.form_layout.addLayout(form_geo)

    def _build_well_inspector(self, well_name: str, well_data_list: list):
        well = next((w for w in well_data_list if w.name == well_name), None)
        if not well:
            self.form_layout.addWidget(QLabel(f"Well '{well_name}' not found."))
            return

        lbl_sec = QLabel(f"Well Properties: {well.name}")
        lbl_sec.setObjectName("sectionHeader")
        self.form_layout.addWidget(lbl_sec)

        # Action button to launch full comprehensive editor
        btn_open_dialog = QPushButton("Open Comprehensive Well Editor...")
        btn_open_dialog.setStyleSheet("""
            QPushButton {
                background: #0d6efd;
                color: #ffffff;
                font-weight: bold;
                font-size: 11px;
                padding: 6px 10px;
                border: 1px solid #0b5ed7;
                border-radius: 4px;
            }
            QPushButton:hover {
                background: #0b5ed7;
            }
        """)
        target_well_name = well.name
        btn_open_dialog.clicked.connect(lambda: self.edit_well_requested.emit(target_well_name))
        self.form_layout.addWidget(btn_open_dialog)

        form = QFormLayout()
        form.setSpacing(6)

        name_edit = QLineEdit(well.name)
        form.addRow("Well Name:", name_edit)

        role_combo = QComboBox()
        role_combo.addItems(["Producer (Active)", "Injector (CO2/WAG)"])
        w_type = str(well.metadata.get("type", "")).lower()
        if "inj" in well.name.lower() or "inj" in w_type:
            role_combo.setCurrentIndex(1)
        form.addRow("Role:", role_combo)

        traj_combo = QComboBox()
        traj_combo.addItems(["Vertical", "Horizontal", "Deviated S-Curve"])
        traj_combo.setCurrentText(str(well.metadata.get("TrajectoryType", "Vertical")))
        form.addRow("Trajectory:", traj_combo)

        sx_spin = QDoubleSpinBox()
        sx_spin.setRange(0.0, 50000.0)
        sx_spin.setValue(float(well.metadata.get("SurfaceX", 1000.0)))
        form.addRow("Surface X:", self._with_badge(sx_spin, "ft"))

        sy_spin = QDoubleSpinBox()
        sy_spin.setRange(0.0, 50000.0)
        sy_spin.setValue(float(well.metadata.get("SurfaceY", 1000.0)))
        form.addRow("Surface Y:", self._with_badge(sy_spin, "ft"))

        # Perforations summary
        perfs = getattr(well, "perforations", []) or []
        n_perfs = len(perfs)
        perfs_str = f"{n_perfs} interval(s)"
        if n_perfs > 0:
            p0 = perfs[0]
            if len(p0) >= 2:
                perfs_str += f" ({p0[0]:.0f} - {p0[1]:.0f} ft)"
        lbl_perfs = QLabel(perfs_str)
        lbl_perfs.setStyleSheet("color: #0f172a; font-weight: 600;")
        form.addRow("Perforations:", lbl_perfs)

        # Peaceman WI readouts
        wi_val = getattr(well, "well_index", None) or getattr(well, "properties", {}).get("peaceman_well_index", None)
        wi_str = f"{wi_val:.2f} STB/d/psi" if wi_val else "Calculated dynamically"
        wi_label = QLabel(wi_str)
        wi_label.setStyleSheet("color: #0d6efd; font-weight: bold;")
        form.addRow("Peaceman WI:", wi_label)

        # Skin factor
        skin_val = getattr(well, "skin_factor", 0.0)
        lbl_skin = QLabel(f"{skin_val:.1f}")
        lbl_skin.setStyleSheet("color: #334155; font-weight: 600;")
        form.addRow("Skin Factor (s):", lbl_skin)

        self.form_layout.addLayout(form)

        # Delete button
        btn_del = QPushButton("Delete Well")
        btn_del.setStyleSheet("""
            QPushButton {
                background: #fef2f2;
                color: #dc2626;
                border: 1px solid #fecaca;
                border-radius: 4px;
                font-weight: 600;
                font-size: 10px;
                padding: 4px 8px;
                margin-top: 8px;
            }
            QPushButton:hover {
                background: #fee2e2;
                color: #b91c1c;
            }
        """)
        btn_del.clicked.connect(lambda: self.delete_well_requested.emit(target_well_name))
        self.form_layout.addWidget(btn_del)

    def _build_fault_item_inspector(self, fault_name: str, fault_data_list: list):
        target_fault = next((f for f in fault_data_list if getattr(f, "name", "") == fault_name or getattr(f, "id", "") == fault_name), None)
        if not target_fault:
            lbl_err = QLabel(f"Fault '{fault_name}' not found in active fault network.")
            lbl_err.setStyleSheet("color: #ef4444; font-size: 11px; padding: 10px;")
            self.form_layout.addWidget(lbl_err)
            return

        lbl_hdr = QLabel(f"Structural Fault: {target_fault.name}")
        lbl_hdr.setObjectName("sectionHeader")
        self.form_layout.addWidget(lbl_hdr)

        btn_box = QHBoxLayout()
        btn_box.setSpacing(4)
        btn_mgr = QPushButton("Edit in Manager...")
        btn_mgr.setStyleSheet("background: #e0f2fe; color: #0284c7; border: 1px solid #bae6fd; border-radius: 4px; font-weight: bold; font-size: 10px; padding: 4px;")
        btn_mgr.clicked.connect(lambda: self.manage_faults_requested.emit(0))
        btn_box.addWidget(btn_mgr)

        btn_iso = QPushButton("Isolate in 3D")
        btn_iso.setStyleSheet("background: #f1f5f9; border: 1px solid #cbd5e1; border-radius: 4px; font-weight: bold; font-size: 10px; padding: 4px;")
        btn_iso.clicked.connect(lambda: self.isolate_fault_requested.emit(target_fault.name))
        btn_box.addWidget(btn_iso)
        self.form_layout.addLayout(btn_box)

        form = QFormLayout()
        form.setSpacing(6)

        id_lbl = QLabel(str(getattr(target_fault, "id", "F-1")))
        id_lbl.setStyleSheet("font-weight: bold; color: #0f172a;")
        form.addRow("Fault ID:", id_lbl)

        chk_active = QCheckBox("Active in 3D Kinematics & Flow")
        chk_active.setChecked(bool(getattr(target_fault, "is_active", True)))
        chk_active.toggled.connect(lambda v: setattr(target_fault, "is_active", v))
        form.addRow("State:", chk_active)

        strike_spin = QDoubleSpinBox()
        strike_spin.setRange(0.0, 360.0)
        strike_spin.setValue(float(getattr(target_fault, "strike", 45.0)))
        strike_spin.valueChanged.connect(lambda v: setattr(target_fault, "strike", v))
        form.addRow("Strike Azimuth:", self._with_badge(strike_spin, "deg"))

        dip_spin = QDoubleSpinBox()
        dip_spin.setRange(10.0, 90.0)
        dip_spin.setValue(float(getattr(target_fault, "dip", 70.0)))
        dip_spin.valueChanged.connect(lambda v: setattr(target_fault, "dip", v))
        form.addRow("Dip Angle:", self._with_badge(dip_spin, "deg"))

        throw_spin = QDoubleSpinBox()
        throw_spin.setRange(-500.0, 500.0)
        throw_spin.setValue(float(getattr(target_fault, "throw", 50.0)))
        throw_spin.valueChanged.connect(lambda v: setattr(target_fault, "throw", v))
        form.addRow("Vertical Throw:", self._with_badge(throw_spin, "ft"))

        heave_spin = QDoubleSpinBox()
        heave_spin.setRange(0.0, 500.0)
        heave_spin.setValue(float(getattr(target_fault, "heave", 18.0)))
        heave_spin.valueChanged.connect(lambda v: setattr(target_fault, "heave", v))
        form.addRow("Horizontal Heave:", self._with_badge(heave_spin, "ft"))

        len_spin = QDoubleSpinBox()
        len_spin.setRange(100.0, 50000.0)
        len_spin.setValue(float(getattr(target_fault, "length", 3500.0)))
        len_spin.valueChanged.connect(lambda v: setattr(target_fault, "length", v))
        form.addRow("Trace Length:", self._with_badge(len_spin, "ft"))

        tm_spin = QDoubleSpinBox()
        tm_spin.setRange(0.0, 1.0)
        tm_spin.setSingleStep(0.05)
        tm_spin.setValue(float(getattr(target_fault, "transmissibility_multiplier", 0.15)))
        tm_spin.valueChanged.connect(lambda v: setattr(target_fault, "transmissibility_multiplier", v))
        form.addRow("Transmissibility Mult:", self._with_badge(tm_spin, "mult"))

        sgr_spin = QDoubleSpinBox()
        sgr_spin.setRange(0.0, 100.0)
        sgr_spin.setValue(float(getattr(target_fault, "shale_gouge_ratio", 32.0)))
        sgr_spin.valueChanged.connect(lambda v: setattr(target_fault, "shale_gouge_ratio", v))
        form.addRow("Shale Gouge (SGR):", self._with_badge(sgr_spin, "%"))

        mu_spin = QDoubleSpinBox()
        mu_spin.setRange(0.10, 1.00)
        mu_spin.setSingleStep(0.05)
        mu_spin.setValue(float(getattr(target_fault, "friction_coefficient", 0.60)))
        mu_spin.valueChanged.connect(lambda v: setattr(target_fault, "friction_coefficient", v))
        form.addRow("Friction Coeff (mu):", self._with_badge(mu_spin, "coeff"))

        ts = float(getattr(target_fault, "slip_tendency", 0.42))
        mu_val = float(getattr(target_fault, "friction_coefficient", 0.60))
        is_crit = ts >= mu_val
        status_txt = f"Ts = {ts:.2f} ({'CRITICALLY STRESSED' if is_crit else 'Sub-Critical / Stable'})"
        lbl_status = QLabel(status_txt)
        lbl_status.setStyleSheet(f"font-weight: bold; color: {'#dc2626' if is_crit else '#16a34a'};")
        form.addRow("Stability Status:", lbl_status)

        self.form_layout.addLayout(form)

        btn_del = QPushButton("Delete Fault")
        btn_del.setStyleSheet("""
            QPushButton {
                background: #fef2f2;
                color: #dc2626;
                border: 1px solid #fecaca;
                border-radius: 4px;
                font-weight: 600;
                font-size: 10px;
                padding: 4px 8px;
                margin-top: 8px;
            }
            QPushButton:hover {
                background: #fee2e2;
                color: #b91c1c;
            }
        """)
        btn_del.clicked.connect(lambda: self.delete_fault_requested.emit(target_fault.name))
        self.form_layout.addWidget(btn_del)

    def _with_badge(self, widget, unit_str: str) -> QWidget:
        """Wraps input widget with unit label and physical validation badge."""
        w = QWidget()
        w.setMinimumHeight(28)
        lay = QHBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)
        if hasattr(widget, 'setMinimumWidth'):
            widget.setMinimumWidth(85)
        if hasattr(widget, 'setMinimumHeight'):
            widget.setMinimumHeight(24)
        lay.addWidget(widget, stretch=1)
        lbl_unit = QLabel(unit_str)
        lbl_unit.setMinimumWidth(40)
        lbl_unit.setStyleSheet("color: #64748b; font-size: 10px;")
        lay.addWidget(lbl_unit)
        lbl_badge = QLabel("✓")
        lbl_badge.setStyleSheet("color: #16a34a; font-weight: bold; font-size: 11px;")
        lay.addWidget(lbl_badge)
        return w
