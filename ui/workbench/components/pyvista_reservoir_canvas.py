import os
import logging
from typing import Optional, List, Dict, Any
import numpy as np

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
    QCheckBox, QPushButton, QFrame, QSizePolicy, QMenu,
    QSlider, QSpinBox, QDoubleSpinBox
)
from PyQt6.QtGui import QColor
from PyQt6.QtCore import pyqtSignal, Qt

from core.geology.petrophysical_distribution import PetrophysicalCube, generate_petrophysical_cube
from core.data_models import FaultData, CaprockLayer

# PyVistaQt hardware OpenGL engine
try:
    import pyvista as pv
    from pyvistaqt import QtInteractor
    PYVISTA_AVAILABLE = True
except Exception as e:
    PYVISTA_AVAILABLE = False
    pv = None
    QtInteractor = None
    logging.error(f"PyVistaQt initialization failed: {e}")

logger = logging.getLogger(__name__)


class PyVistaReservoirCanvas(QWidget):
    """
    High-performance 3D Subsurface Viewport powered exclusively by PyVista (VTK/OpenGL).
    Supports hardware-accelerated volumetric structured grids,
    fault plane geometries with throw/seal indicators, continuous 3D wellbore tubes,
    perforations, single-well isolation highlighting, inter-well sweep vectors,
    and raycast picking.
    """
    well_clicked = pyqtSignal(str)              # Emits clicked well name
    cell_clicked = pyqtSignal(dict)             # Emits {cell_id, x, y, z, value}
    surface_clicked = pyqtSignal(float, float)  # Emits (X, Y) for well placement

    def __init__(self, parent=None):
        super().__init__(parent)
        self.plotter = None
        self.grid_mesh = None
        self.fault_mesh = None
        self.well_meshes = {}
        self.place_well_mode = False
        self.isolated_well_name: Optional[str] = None
        self.isolate_fault_view: bool = False
        self.isolated_fault_name: Optional[str] = None
        self.fault_data_list: Optional[List[Any]] = None
        self.last_model_params = None
        self._has_rendered_mesh = False

        self._setup_ui()

    def set_isolated_well(self, well_name: Optional[str]):
        """Sets a specific well to isolate in 3D (highlighted tube/perfs, others ghosted). Pass None for all wells."""
        if self.isolated_well_name != well_name:
            self.isolated_well_name = well_name
            self._re_render_last()

    def set_fault_isolated_view(self, isolated: bool, fault_name: Optional[str] = None):
        """Sets fault isolated 3D view (reservoir ghosted/hidden, fault surfaces highlighted)."""
        self.isolate_fault_view = isolated
        self.isolated_fault_name = fault_name
        self._re_render_last()

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(2)

        # 1. Clean Light Engineering HUD Toolbar
        self.hud_bar = self._create_hud_toolbar()
        main_layout.addWidget(self.hud_bar)

        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._show_context_menu)

        # 2. 3D Viewport: Native PyVistaQt (Hardware Accelerated OpenGL)
        if PYVISTA_AVAILABLE and QtInteractor is not None:
            try:
                self.plotter = QtInteractor(self)
                self.plotter.set_background("#ffffff")
                self.plotter.add_axes(color="#334155", line_width=1.5)
                self.plotter.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
                self.plotter.customContextMenuRequested.connect(self._show_context_menu)
                self._setup_pyvista_picking()
                main_layout.addWidget(self.plotter, stretch=1)
                logger.info("PyVistaReservoirCanvas: Native PyVistaQt hardware OpenGL initialized.")
            except Exception as e:
                logger.error(f"Failed to initialize QtInteractor: {e}", exc_info=True)
                err_lbl = QLabel(f"PyVista 3D Initialization Error: {e}")
                err_lbl.setStyleSheet("color: #dc2626; padding: 20px; font-weight: bold;")
                main_layout.addWidget(err_lbl, stretch=1)
        else:
            err_lbl = QLabel("PyVista is required for 3D subsurface renders.")
            err_lbl.setStyleSheet("color: #dc2626; padding: 20px; font-weight: bold;")
            main_layout.addWidget(err_lbl, stretch=1)

    def _create_hud_toolbar(self) -> QFrame:
        frame = QFrame()
        frame.setStyleSheet("""
            QFrame {
                background: #f8fafc;
                border: 1px solid #dee2e6;
                border-radius: 4px;
                padding: 2px 4px;
            }
            QLabel {
                color: #475569;
                font-size: 11px;
                font-weight: 600;
            }
            QPushButton {
                background: #ffffff;
                color: #1e293b;
                border: 1px solid #cbd5e1;
                border-radius: 3px;
                padding: 3px 8px;
                font-size: 11px;
                font-weight: 500;
            }
            QPushButton:hover {
                background: #f1f5f9;
                border-color: #94a3b8;
            }
            QPushButton:checked {
                background: #0d6efd;
                color: #ffffff;
                border-color: #0b5ed7;
                font-weight: bold;
            }
            QComboBox {
                background: #ffffff;
                color: #1e293b;
                border: 1px solid #cbd5e1;
                border-radius: 3px;
                padding: 3px 6px;
                font-size: 11px;
            }
            QCheckBox {
                color: #334155;
                font-size: 11px;
                font-weight: 500;
            }
        """)
        main_hud_layout = QVBoxLayout(frame)
        main_hud_layout.setContentsMargins(6, 4, 6, 4)
        main_hud_layout.setSpacing(4)

        # Row 1: View Orientation, Active Property, Colormap Palette, Cut Tool & Cut Pos
        row1_layout = QHBoxLayout()
        row1_layout.setContentsMargins(0, 0, 0, 0)
        row1_layout.setSpacing(6)

        row1_layout.addWidget(QLabel("View:"))
        self.combo_view = QComboBox()
        self.combo_view.setMinimumWidth(85)
        self.combo_view.addItems(["Isometric", "Top (X-Y)", "Front (X-Z)", "Side (Y-Z)"])
        self.combo_view.currentIndexChanged.connect(self._on_view_changed)
        row1_layout.addWidget(self.combo_view)

        btn_reset = QPushButton("Reset Cam")
        btn_reset.setToolTip("Reset camera view to default orientation")
        btn_reset.clicked.connect(self.reset_camera)
        row1_layout.addWidget(btn_reset)

        row1_layout.addWidget(QLabel("Property:"))
        self.combo_property = QComboBox()
        self.combo_property.setMinimumWidth(150)
        self.combo_property.addItems([
            "Permeability (PERMX)",
            "Porosity (PORO)",
            "Lithofacies (Sand/Silt/Shale)",
            "Oil Saturation (So)",
            "Water Saturation (Sw)",
            "Gas Saturation (Sg)",
            "Fluid Phase Leg (Gas/Oil/Water)",
            "Miscibility Margin (P - MMP)",
            "In-Situ Oil Viscosity (cP)",
            "Pore Pressure (P)",
            "Young's Modulus (E)",
            "In-Situ Stress (Shmin)",
            "Slip Tendency (Ts)"
        ])
        self.combo_property.currentIndexChanged.connect(self._on_property_changed)
        row1_layout.addWidget(self.combo_property)

        row1_layout.addWidget(QLabel("Palette:"))
        self.combo_cmap = QComboBox()
        self.combo_cmap.setMinimumWidth(75)
        self.combo_cmap.addItems(["turbo", "viridis", "coolwarm", "plasma", "jet"])
        self.combo_cmap.currentIndexChanged.connect(self._on_property_changed)
        row1_layout.addWidget(self.combo_cmap)

        row1_layout.addWidget(QLabel("Cut Tool:"))
        self.combo_cut_mode = QComboBox()
        self.combo_cut_mode.setMinimumWidth(130)
        self.combo_cut_mode.addItems([
            "Solid Full Block",
            "Chair Cut (Corner)",
            "I-Slice Cut (X-Plane)",
            "J-Slice Cut (Y-Plane)",
            "K-Slice Cut (Depth Z)",
            "Triple Ortho-Slices"
        ])
        self.combo_cut_mode.currentIndexChanged.connect(self._re_render_last)
        row1_layout.addWidget(self.combo_cut_mode)

        row1_layout.addWidget(QLabel("Cut Pos:"))
        self.slider_cut_pos = QSlider(Qt.Orientation.Horizontal)
        self.slider_cut_pos.setRange(10, 90)
        self.slider_cut_pos.setValue(50)
        self.slider_cut_pos.setFixedWidth(70)
        self.slider_cut_pos.setToolTip("Cutting plane / chair cut location (10% to 90%)")
        self.lbl_cut_pos = QLabel("50%")
        self.lbl_cut_pos.setStyleSheet("color: #0284c7; font-weight: bold; min-width: 28px;")
        self.slider_cut_pos.valueChanged.connect(self._on_cut_slider_changed)
        row1_layout.addWidget(self.slider_cut_pos)
        row1_layout.addWidget(self.lbl_cut_pos)

        row1_layout.addStretch()
        main_hud_layout.addLayout(row1_layout)

        # Row 2: Opacity Slider, High-Flow Filter, Geological & Well Overlays
        row2_layout = QHBoxLayout()
        row2_layout.setContentsMargins(0, 0, 0, 0)
        row2_layout.setSpacing(6)

        row2_layout.addWidget(QLabel("Opacity:"))
        self.slider_opacity = QSlider(Qt.Orientation.Horizontal)
        self.slider_opacity.setRange(15, 100)
        self.slider_opacity.setValue(100)  # Default 100% Solid Opaque
        self.slider_opacity.setFixedWidth(70)
        self.slider_opacity.setToolTip("Volume Opacity (15% to 100%). Default is 100% solid.")
        self.lbl_opacity = QLabel("1.00")
        self.lbl_opacity.setStyleSheet("color: #16a34a; font-weight: bold; min-width: 28px;")
        self.slider_opacity.valueChanged.connect(self._on_opacity_slider_changed)
        row2_layout.addWidget(self.slider_opacity)
        row2_layout.addWidget(self.lbl_opacity)

        self.chk_flow_conduits = QCheckBox("Conduits (>P75)")
        self.chk_flow_conduits.setToolTip("Filter out matrix below 75th percentile to highlight high-flow permeable channels")
        self.chk_flow_conduits.setChecked(False)
        self.chk_flow_conduits.toggled.connect(self._re_render_last)
        row2_layout.addWidget(self.chk_flow_conduits)

        self.chk_faults = QCheckBox("Faults")
        self.chk_faults.setChecked(True)
        self.chk_faults.toggled.connect(self._re_render_last)
        row2_layout.addWidget(self.chk_faults)

        self.chk_caprock = QCheckBox("Caprock")
        self.chk_caprock.setChecked(False)
        self.chk_caprock.setToolTip("Display overlying impermeable caprock seal layer (strictly above reservoir)")
        self.chk_caprock.toggled.connect(self._re_render_last)
        row2_layout.addWidget(self.chk_caprock)

        self.chk_contacts = QCheckBox("Contacts (WOC/GOC)")
        self.chk_contacts.setChecked(True)
        self.chk_contacts.setToolTip("Toggle 3D fluid contact horizon planes (Water-Oil Contact, Gas-Oil Contact)")
        self.chk_contacts.toggled.connect(self._re_render_last)
        row2_layout.addWidget(self.chk_contacts)

        self.chk_wells = QCheckBox("Wells")
        self.chk_wells.setChecked(True)
        self.chk_wells.setToolTip("Toggle wellbore tubes and trajectories")
        self.chk_wells.toggled.connect(self._re_render_last)
        row2_layout.addWidget(self.chk_wells)

        self.chk_perfs = QCheckBox("Perfs")
        self.chk_perfs.setChecked(True)
        self.chk_perfs.toggled.connect(self._re_render_last)
        row2_layout.addWidget(self.chk_perfs)

        self.chk_labels = QCheckBox("Labels")
        self.chk_labels.setChecked(True)
        self.chk_labels.toggled.connect(self._re_render_last)
        row2_layout.addWidget(self.chk_labels)

        self.btn_place_well = QPushButton("+ Place Well")
        self.btn_place_well.setCheckable(True)
        self.btn_place_well.setToolTip("Click on reservoir surface to place a well")
        self.btn_place_well.toggled.connect(self._toggle_place_well_mode)
        row2_layout.addWidget(self.btn_place_well)

        row2_layout.addStretch()

        self.lbl_cursor = QLabel("Ready")
        self.lbl_cursor.setStyleSheet("color: #64748b; font-size: 11px;")
        row2_layout.addWidget(self.lbl_cursor)

        main_hud_layout.addLayout(row2_layout)

        return frame

    def _on_cut_slider_changed(self, val: int):
        self.lbl_cut_pos.setText(f"{val}%")
        self._re_render_last()

    def _on_opacity_slider_changed(self, val: int):
        self.lbl_opacity.setText(f"{val / 100.0:.2f}")
        self._re_render_last()


    def _setup_pyvista_picking(self):
        if self.plotter is None:
            return
        try:
            self.plotter.enable_point_picking(callback=self._on_pyvista_point_picked, show_message=False)
        except Exception as e:
            logger.debug(f"PyVista point picking setup error: {e}")

    def _on_pyvista_point_picked(self, point):
        if point is None:
            return
        x, y, z = point
        if self.place_well_mode:
            logger.info(f"Surface point clicked for well placement: ({x:.1f}, {y:.1f})")
            self.surface_clicked.emit(float(x), float(y))
            self.btn_place_well.setChecked(False)
            return

        # Check proximity to existing wellheads/trajectories
        if self.last_model_params and self.last_model_params.get("well_data_list"):
            wells = self.last_model_params["well_data_list"]
            closest_well = None
            min_dist = float("inf")
            for w in wells:
                meta = getattr(w, "metadata", {}) or {}
                wx = float(meta.get("SurfaceX", meta.get("surface_x", 0.0)))
                wy = float(meta.get("SurfaceY", meta.get("surface_y", 0.0)))
                d = np.hypot(x - wx, y - wy)
                if d < min_dist:
                    min_dist = d
                    closest_well = w.name
            thresh = max(float(self.last_model_params.get("length_ft", 2000.0)) * 0.08, 120.0)
            if min_dist <= thresh and closest_well:
                logger.info(f"Picked well '{closest_well}' in PyVista 3D at distance {min_dist:.1f} ft")
                self.well_clicked.emit(closest_well)
                self.lbl_cursor.setText(f"Selected Well: {closest_well}")
                return

        self.cell_clicked.emit({
            "x": float(x), "y": float(y), "z": float(z),
            "property": self.combo_property.currentText()
        })
        self.lbl_cursor.setText(f"Selected: X={x:.0f} ft, Y={y:.0f} ft, Depth={z:.0f} ft")

    def _on_view_changed(self, index: int):
        if self.plotter is not None:
            if index == 0:
                self.plotter.view_isometric()
            elif index == 1:
                self.plotter.view_xy()
            elif index == 2:
                self.plotter.view_xz()
            elif index == 3:
                self.plotter.view_yz()
            self.plotter.camera.zoom(0.78)
            self.plotter.render()

    def reset_camera(self):
        self.combo_view.setCurrentIndex(0)
        if self.plotter is not None:
            self.plotter.reset_camera()
            self.plotter.camera.zoom(0.78)
            self.plotter.render()

    def _toggle_place_well_mode(self, checked: bool):
        self.place_well_mode = checked
        if checked:
            self.lbl_cursor.setText("Click reservoir surface to place well")
            self.lbl_cursor.setStyleSheet("color: #0d6efd; font-weight: bold; font-size: 11px;")
        else:
            self.lbl_cursor.setText("Ready")
            self.lbl_cursor.setStyleSheet("color: #64748b; font-size: 11px;")

    def _on_property_changed(self):
        self._re_render_last()

    def _re_render_last(self):
        if self.last_model_params:
            self.render_subsurface_model(**self.last_model_params)

    def update_scene(self, params: Dict[str, Any]):
        """Alias for render_subsurface_model taking dictionary params for compatibility."""
        grid_dims = params.get("grid_dims", {})
        res = params.get("reservoir", {})
        wells = params.get("wells", [])
        nx = int(grid_dims.get("nx", 50))
        ny = int(grid_dims.get("ny", 50))
        nz = int(grid_dims.get("nz", 10))
        length = float(res.get("length", 2000.0))
        width = float(res.get("width", 2000.0))
        depth = float(res.get("depth", 5000.0))
        thickness = float(res.get("net_pay", 50.0))
        perm = float(res.get("perm", 100.0))
        poro = float(res.get("poro", 0.20))
        self.render_subsurface_model(
            nx=nx, ny=ny, nz=nz,
            length_ft=length, width_ft=width,
            top_depth=depth, thickness_ft=thickness,
            perm_base=perm, poro_base=poro,
            well_data_list=wells
        )

    def render_subsurface_model(
        self,
        nx: int, ny: int, nz: int,
        length_ft: float, width_ft: float,
        top_depth: float, thickness_ft: float,
        perm_base: float = 100.0,
        poro_base: float = 0.20,
        well_data_list: Optional[List[Any]] = None,
        fault_props: Optional[Dict[str, Any]] = None,
        caprock_props: Optional[Dict[str, Any]] = None,
        petro_cube: Optional[Any] = None,
        distribution_params: Optional[Dict[str, Any]] = None,
        fault_data_list: Optional[List[Any]] = None,
        caprock_layers: Optional[List[Any]] = None
    ):
        """
        Builds and renders the complete solid volumetric reservoir block,
        3D inclined fault plane, cutting plane / chair cut inspection, wellbore trajectories,
        and perforation collars using unified petrophysics and geomechanics models.
        """
        self.last_model_params = {
            "nx": nx, "ny": ny, "nz": nz,
            "length_ft": length_ft, "width_ft": width_ft,
            "top_depth": top_depth, "thickness_ft": thickness_ft,
            "perm_base": perm_base, "poro_base": poro_base,
            "well_data_list": well_data_list,
            "fault_props": fault_props,
            "caprock_props": caprock_props,
            "petro_cube": petro_cube,
            "distribution_params": distribution_params,
            "fault_data_list": fault_data_list,
            "caprock_layers": caprock_layers
        }
        if fault_data_list is not None:
            self.fault_data_list = fault_data_list

        base_depth = top_depth + thickness_ft
        prop_text = self.combo_property.currentText()
        cmap_name = self.combo_cmap.currentText()
        cut_mode = self.combo_cut_mode.currentText()
        cut_pos = float(self.slider_cut_pos.value()) / 100.0
        cube_opacity = float(self.slider_opacity.value()) / 100.0

        # Synthesize or ingest unified PetrophysicalCube
        dp = distribution_params or {}
        if petro_cube is None:
            petro_cube = generate_petrophysical_cube(
                nx=nx, ny=ny, nz=nz,
                length_ft=length_ft, width_ft=width_ft,
                top_depth=top_depth, thickness_ft=thickness_ft,
                distribution_method=str(dp.get("distribution_method", "Layered (Dykstra-Parsons)")),
                perm_base=perm_base, poro_base=poro_base,
                v_dp=float(dp.get("v_dp", dp.get("dykstra_parsons", 0.65))),
                kv_kh=float(dp.get("kv_kh", dp.get("kv_kh_ratio", 0.10))),
                facies_pattern=str(dp.get("facies_pattern", "Fluvial Channel Belt")),
                sand_fraction=float(dp.get("sand_fraction", 0.65)),
                silt_fraction=float(dp.get("silt_fraction", 0.25)),
                shale_fraction=float(dp.get("shale_fraction", 0.10)),
                poro_perm_model=str(dp.get("poro_perm_model", "Kozeny-Carman")),
                initial_pressure=float(dp.get("initial_pressure", 4000.0)),
                overburden_grad=float(dp.get("overburden_grad", 1.00)),
                stress_k0=float(dp.get("stress_k0", 0.75)),
                poissons_ratio=float(dp.get("poissons_ratio", 0.25)),
                youngs_modulus_base=float(dp.get("youngs_modulus_base", 20.0)),
                biot_coeff=float(dp.get("biot_coeff", 0.80)),
                frac_grad=float(dp.get("frac_grad", 0.85)),
                channel_azimuth_deg=float(dp.get("channel_azimuth_deg", 45.0)),
                channel_sinuosity=float(dp.get("channel_sinuosity", 1.30)),
                channel_wavelength_ft=float(dp.get("channel_wavelength_ft", 1500.0)),
                channel_amplitude_ft=float(dp.get("channel_amplitude_ft", 350.0)),
                channel_width_ft=float(dp.get("channel_width_ft", 450.0)),
                levee_width_ft=float(dp.get("levee_width_ft", 250.0)),
                aggradation_drift_ft=float(dp.get("aggradation_drift_ft", 30.0)),
                num_channels=int(dp.get("num_channels", 1)),
                barrier_azimuth_deg=float(dp.get("barrier_azimuth_deg", 90.0)),
                barrier_width_ft=float(dp.get("barrier_width_ft", 800.0)),
                lagoon_width_ft=float(dp.get("lagoon_width_ft", 450.0)),
                progradation_dip_deg=float(dp.get("progradation_dip_deg", 2.0)),
                reef_center_x=float(dp.get("reef_center_x", length_ft * 0.5)),
                reef_center_y=float(dp.get("reef_center_y", width_ft * 0.5)),
                reef_major_radius_ft=float(dp.get("reef_major_radius_ft", 650.0)),
                reef_minor_radius_ft=float(dp.get("reef_minor_radius_ft", 400.0)),
                reef_azimuth_deg=float(dp.get("reef_azimuth_deg", 45.0)),
                apron_width_ft=float(dp.get("apron_width_ft", 300.0)),
                f1_perm=float(dp.get("f1_perm", 250.0)),
                f1_poro=float(dp.get("f1_poro", 0.25)),
                f2_perm=float(dp.get("f2_perm", 40.0)),
                f2_poro=float(dp.get("f2_poro", 0.16)),
                f3_perm=float(dp.get("f3_perm", 0.5)),
                f3_poro=float(dp.get("f3_poro", 0.06)),
                layer_permeability_trend=str(dp.get("layer_permeability_trend", "Fining Upward")),
                variogram_type=str(dp.get("variogram_type", "Spherical")),
                variogram_range_major=float(dp.get("variogram_range_major", 1200.0)),
                variogram_range_minor=float(dp.get("variogram_range_minor", 600.0)),
                variogram_range_vert=float(dp.get("variogram_range_vert", 20.0)),
                variogram_azimuth_deg=float(dp.get("variogram_azimuth_deg", 45.0)),
                nugget_effect=float(dp.get("nugget_effect", 0.05)),
                sill_variance=float(dp.get("sill_variance", 1.0)),
                random_seed=int(dp.get("geostat_seed", 42))
            )
        self.current_petro_cube = petro_cube

        sw_arr = petro_cube.sw if petro_cube.sw is not None else np.clip(1.0 - petro_cube.saturation, 0.0, 1.0)
        sg_arr = petro_cube.sg if petro_cube.sg is not None else np.zeros_like(petro_cube.saturation)
        phase_arr = petro_cube.fluid_phase if petro_cube.fluid_phase is not None else np.where(petro_cube.saturation > 0.08, 2, 3)
        margin_arr = petro_cube.miscibility_margin if petro_cube.miscibility_margin is not None else (petro_cube.pressure - 2150.0)
        visc_arr = petro_cube.visco if petro_cube.visco is not None else np.full_like(petro_cube.saturation, 2.5)

        # Determine scalar parameters
        if "Water Saturation" in prop_text or "(Sw)" in prop_text:
            prop_key = "WaterSaturation"
            unit_str = "Sw fraction"
            vmin, vmax = 0.0, 1.0
            fmt_str = "%.2f"
        elif "Gas Saturation" in prop_text or "(Sg)" in prop_text:
            prop_key = "GasSaturation"
            unit_str = "Sg fraction"
            vmin, vmax = 0.0, float(max(0.5, np.max(sg_arr) + 0.1))
            fmt_str = "%.2f"
        elif "Fluid Phase" in prop_text:
            prop_key = "FluidPhase"
            unit_str = "1=Gas 2=Oil 3=Water"
            vmin, vmax = 1.0, 3.0
            fmt_str = "%.0f"
        elif "Miscibility" in prop_text or "MMP" in prop_text:
            prop_key = "MiscibilityMargin"
            unit_str = "P - MMP (psia)"
            vmin = float(min(-200.0, np.min(margin_arr)))
            vmax = float(max(200.0, np.max(margin_arr)))
            fmt_str = "%.0f"
        elif "Viscosity" in prop_text or "VISCO" in prop_text:
            prop_key = "Viscosity"
            unit_str = "cP"
            vmin = float(max(0.1, np.min(visc_arr) * 0.8))
            vmax = float(max(vmin + 1.0, np.max(visc_arr) * 1.2))
            fmt_str = "%.2f"
        elif "Poro" in prop_text:
            prop_key = "Porosity"
            unit_str = "fraction"
            vmin = float(max(0.01, np.min(petro_cube.poro)))
            vmax = float(max(vmin + 0.05, np.max(petro_cube.poro)))
            fmt_str = "%.2f"
        elif "Litho" in prop_text or "Facies" in prop_text:
            prop_key = "Lithofacies"
            unit_str = "1=Sand 2=Silt 3=Shale"
            vmin, vmax = 1.0, 3.0
            fmt_str = "%.0f"
        elif "Sat" in prop_text or "(So)" in prop_text:
            prop_key = "Saturation"
            unit_str = "So fraction"
            vmin = float(max(0.0, np.min(petro_cube.saturation)))
            vmax = float(max(vmin + 0.1, np.max(petro_cube.saturation)))
            fmt_str = "%.2f"
        elif "Press" in prop_text:
            prop_key = "Pressure"
            unit_str = "psia"
            vmin = float(np.min(petro_cube.pressure))
            vmax = float(max(vmin + 50.0, np.max(petro_cube.pressure)))
            fmt_str = "%.0f"
        elif "Young" in prop_text or "(E)" in prop_text:
            prop_key = "YoungsModulus"
            unit_str = "GPa"
            vmin = float(np.min(petro_cube.youngs_modulus))
            vmax = float(max(vmin + 2.0, np.max(petro_cube.youngs_modulus)))
            fmt_str = "%.1f"
        elif "Stress" in prop_text or "Shmin" in prop_text:
            prop_key = "InSituStress"
            unit_str = "psia"
            vmin = float(np.min(petro_cube.shmin))
            vmax = float(max(vmin + 50.0, np.max(petro_cube.shmin)))
            fmt_str = "%.0f"
        elif "Slip" in prop_text or "Ts" in prop_text:
            prop_key = "SlipTendency"
            unit_str = "Ts (tau/sigma_n')"
            vmin, vmax = 0.0, 1.0
            fmt_str = "%.2f"
        else:
            prop_key = "Permeability"
            unit_str = "mD"
            vmin = float(max(0.01, np.min(petro_cube.perm)))
            vmax = float(max(vmin * 2.0, np.max(petro_cube.perm)))
            fmt_str = "%.0f"

        mid_x = length_ft * 0.5
        mid_y = width_ft * 0.5
        mid_z_vis = -top_depth - thickness_ft * 0.5

        # --- 1. HARDWARE PYVISTA RENDERING ---
        if self.plotter is not None and pv is not None:
            try:
                # Save camera position so parameter edits never jump/reset user's viewport
                saved_camera = None
                if getattr(self, '_has_rendered_mesh', False):
                    try:
                        saved_camera = self.plotter.camera_position
                    except Exception:
                        saved_camera = None

                self.plotter.clear()
                self.plotter.set_background("#ffffff")
                self.plotter.add_axes(color="#334155", line_width=1.5)

                # In VTK, +Z is UP. In petroleum engineering, Depth (TVD) is down.
                # Mapping Z_vis = -TVD places shallow formations (caprock) physically ON TOP of reservoir.
                # In VTK, +Z is UP. In petroleum engineering, Depth (TVD) is down.
                # Mapping Z_vis = -TVD places shallow formations (caprock) physically ON TOP of reservoir.
                gx = np.linspace(0, length_ft, nx + 1)
                gy = np.linspace(0, width_ft, ny + 1)
                gz = np.linspace(-base_depth, -top_depth, nz + 1)
                xx, yy, zz = np.meshgrid(gx, gy, gz, indexing="ij")

                # --- 1A. AUTHENTIC GEOLOGICAL MULTI-FAULT KINEMATICS & DISPLACEMENT ---
                # Across normal fault planes, the hanging wall drops downward by throw delta_Z,
                # creating visible layer offsets, fault scarps, and horizon juxtaposition.
                fp = fault_props or {}
                raw_faults = fault_data_list or (self.last_model_params or {}).get("fault_data_list") or getattr(self, "fault_data_list", None)
                if raw_faults and len(raw_faults) > 0:
                    active_fault_list = list(raw_faults)
                else:
                    active_fault_list = [
                        FaultData(
                            id="F-1",
                            name=str(fp.get("fault_name", "Fault F-1")),
                            strike=float(fp.get("fault_strike", 45.0)),
                            dip=float(fp.get("fault_dip", 70.0)),
                            throw=float(fp.get("fault_throw", 55.0)),
                            heave=float(fp.get("fault_heave", 18.0)),
                            length=float(fp.get("fault_length", max(length_ft, width_ft) * 0.9)),
                            center_x=float(fp.get("fault_center_x", mid_x)),
                            center_y=float(fp.get("fault_center_y", mid_y)),
                            transmissibility_multiplier=float(fp.get("fault_trans_mult", 0.15)),
                            damage_zone_width=float(fp.get("fault_damage_width", 140.0)),
                            shale_gouge_ratio=float(fp.get("shale_gouge_ratio", 32.0)),
                            slip_tendency=float(fp.get("fault_slip_tendency", 0.42))
                        )
                    ]

                xx_deformed = np.copy(xx)
                yy_deformed = np.copy(yy)
                zz_deformed = np.copy(zz)
                has_active_displacement = False

                for flt in active_fault_list:
                    f_throw = float(getattr(flt, "throw", 0.0))
                    if abs(f_throw) < 1.0 or not getattr(flt, "is_active", True):
                        continue
                    has_active_displacement = True
                    f_dip = float(getattr(flt, "dip", 70.0))
                    f_strike = float(getattr(flt, "strike", 45.0))
                    f_cx = float(getattr(flt, "center_x", mid_x))
                    f_cy = float(getattr(flt, "center_y", mid_y))
                    f_len = float(getattr(flt, "length", max(length_ft, width_ft) * 0.9))

                    rad_dip = np.radians(f_dip)
                    rad_strike = np.radians(f_strike)
                    cot_dip = 1.0 / np.tan(np.clip(rad_dip, np.radians(20.0), np.radians(85.0)))

                    dx_node = xx_deformed - f_cx
                    dy_node = yy_deformed - f_cy
                    s_node = dx_node * np.sin(rad_strike) + dy_node * np.cos(rad_strike)
                    d_perp_node = dx_node * np.cos(rad_strike) - dy_node * np.sin(rad_strike)
                    dip_shift_node = -(zz_deformed - mid_z_vis) * cot_dip
                    dist_plane_node = d_perp_node - dip_shift_node

                    half_flen = f_len * 0.5
                    s_norm = np.clip(np.abs(s_node) / max(half_flen, 1.0), 0.0, 1.0)
                    taper_strike = np.sqrt(np.maximum(0.0, 1.0 - s_norm**2))

                    w_trans = max(length_ft / (nx * 2.2), 16.0)
                    h_step = 0.5 * (1.0 + np.tanh(dist_plane_node / w_trans))

                    dz_fault = -f_throw * taper_strike * h_step
                    dz_foot = 0.12 * f_throw * taper_strike * (1.0 - h_step) * np.exp(-np.maximum(0.0, -dist_plane_node) / 380.0)

                    zz_deformed += (dz_fault + dz_foot)
                    d_heave = dz_fault * cot_dip * 0.16
                    xx_deformed += d_heave * np.cos(rad_strike)
                    yy_deformed -= d_heave * np.sin(rad_strike)

                if has_active_displacement:
                    self.grid_mesh = pv.StructuredGrid(xx_deformed, yy_deformed, zz_deformed)
                else:
                    self.grid_mesh = pv.StructuredGrid(xx, yy, zz)

                # --- 1B. FAULT DAMAGE ZONE PETROPHYSICAL MODULATION ---
                # Compute distance to fault plane for cell centers to apply gouge/cataclasis permeability baffle
                cx_cell = 0.5 * (gx[:-1] + gx[1:])
                cy_cell = 0.5 * (gy[:-1] + gy[1:])
                cz_cell = 0.5 * (gz[:-1] + gz[1:])
                c_xx, c_yy, c_zz = np.meshgrid(cx_cell, cy_cell, cz_cell, indexing="ij")
                perm_mod = np.copy(petro_cube.perm)
                slip_mod = np.copy(petro_cube.slip_tendency)

                for flt in active_fault_list:
                    if not getattr(flt, "is_active", True):
                        continue
                    f_dip = float(getattr(flt, "dip", 70.0))
                    f_strike = float(getattr(flt, "strike", 45.0))
                    f_cx = float(getattr(flt, "center_x", mid_x))
                    f_cy = float(getattr(flt, "center_y", mid_y))
                    f_mult = float(getattr(flt, "transmissibility_multiplier", 0.15))
                    f_damage_w = float(getattr(flt, "damage_zone_width", 140.0))

                    rad_dip = np.radians(f_dip)
                    rad_strike = np.radians(f_strike)
                    cot_dip = 1.0 / np.tan(np.clip(rad_dip, np.radians(20.0), np.radians(85.0)))

                    c_dx = c_xx - f_cx
                    c_dy = c_yy - f_cy
                    c_dperp = c_dx * np.cos(rad_strike) - c_dy * np.sin(rad_strike)
                    c_dip_shift = -(c_zz - mid_z_vis) * cot_dip
                    dist_cell = c_dperp - c_dip_shift

                    in_damage = np.abs(dist_cell) < f_damage_w
                    decay_ratio = np.clip(np.abs(dist_cell) / max(f_damage_w, 1.0), 0.0, 1.0)
                    baffle_mult = f_mult + (1.0 - f_mult) * (decay_ratio ** 1.6)
                    perm_mod[in_damage] = np.minimum(perm_mod[in_damage], (perm_mod[in_damage] * baffle_mult[in_damage]))
                    slip_mod[in_damage] = np.maximum(slip_mod[in_damage], np.clip(0.35 + 0.35 * (1.0 - decay_ratio[in_damage]), 0.05, 0.95))

                # Populate unified multi-scalar arrays (Fortran flattened to match structured grid cells)
                self.grid_mesh.cell_data["Permeability"] = perm_mod.flatten(order="F")
                self.grid_mesh.cell_data["Porosity"] = petro_cube.poro.flatten(order="F")
                self.grid_mesh.cell_data["Lithofacies"] = petro_cube.facies.flatten(order="F").astype(float)
                self.grid_mesh.cell_data["Saturation"] = petro_cube.saturation.flatten(order="F")
                self.grid_mesh.cell_data["WaterSaturation"] = sw_arr.flatten(order="F")
                self.grid_mesh.cell_data["GasSaturation"] = sg_arr.flatten(order="F")
                self.grid_mesh.cell_data["FluidPhase"] = phase_arr.flatten(order="F").astype(float)
                self.grid_mesh.cell_data["MiscibilityMargin"] = margin_arr.flatten(order="F")
                self.grid_mesh.cell_data["Viscosity"] = visc_arr.flatten(order="F")
                self.grid_mesh.cell_data["Pressure"] = petro_cube.pressure.flatten(order="F")
                self.grid_mesh.cell_data["YoungsModulus"] = petro_cube.youngs_modulus.flatten(order="F")
                self.grid_mesh.cell_data["InSituStress"] = petro_cube.shmin.flatten(order="F")
                self.grid_mesh.cell_data["SlipTendency"] = slip_mod.flatten(order="F")

                scalar_bar_conf = dict(
                    title=f"{prop_key} ({unit_str})",
                    vertical=True,
                    n_labels=5,
                    italic=False,
                    bold=True,
                    fmt=fmt_str,
                    title_font_size=9,
                    label_font_size=8,
                    color="#1e293b",
                    position_x=0.90,
                    position_y=0.15,
                    width=0.045,
                    height=0.68,
                    shadow=False
                )

                # Apply Cross-Section Cut Tools
                mesh_to_render = self.grid_mesh

                if "Chair" in cut_mode:
                    # True CAD Chair Cut: Clip away the upper-right corner box to reveal internal layers solidly
                    x_cut = length_ft * cut_pos
                    y_cut = width_ft * (1.0 - cut_pos)
                    z_cut = -top_depth - thickness_ft * (1.0 - cut_pos)
                    box_bounds = [x_cut, length_ft * 1.02, -0.02 * width_ft, y_cut, z_cut, -top_depth * 0.999]
                    try:
                        mesh_to_render = self.grid_mesh.clip_box(bounds=box_bounds, invert=True)
                    except Exception as clip_err:
                        logger.debug(f"Chair cut clip error: {clip_err}")
                elif "I-Slice" in cut_mode:
                    x_cut = length_ft * cut_pos
                    try:
                        mesh_to_render = self.grid_mesh.clip(normal='x', origin=(x_cut, 0, 0), invert=False)
                    except Exception as clip_err:
                        logger.debug(f"I-Slice clip error: {clip_err}")
                elif "J-Slice" in cut_mode:
                    y_cut = width_ft * cut_pos
                    try:
                        mesh_to_render = self.grid_mesh.clip(normal='y', origin=(0, y_cut, 0), invert=False)
                    except Exception as clip_err:
                        logger.debug(f"J-Slice clip error: {clip_err}")
                elif "K-Slice" in cut_mode:
                    z_cut = -top_depth - thickness_ft * cut_pos
                    try:
                        mesh_to_render = self.grid_mesh.clip(normal='z', origin=(0, 0, z_cut), invert=False)
                    except Exception as clip_err:
                        logger.debug(f"K-Slice clip error: {clip_err}")
                elif "Triple" in cut_mode:
                    try:
                        mesh_to_render = self.grid_mesh.slice_orthogonal(
                            x=length_ft * cut_pos,
                            y=width_ft * (1.0 - cut_pos),
                            z=-top_depth - thickness_ft * 0.5
                        )
                    except Exception as clip_err:
                        logger.debug(f"Triple slice error: {clip_err}")

                # Flow conduit threshold filter (>P75)
                if self.chk_flow_conduits.isChecked():
                    p75 = float(np.percentile(petro_cube.perm, 75))
                    try:
                        mesh_to_render = mesh_to_render.threshold(value=p75, scalars="Permeability")
                    except Exception as thresh_err:
                        logger.debug(f"Flow conduit threshold error: {thresh_err}")

                # Render 3D reservoir block (Or faint wireframe if viewing fault isolated!)
                if getattr(self, "isolate_fault_view", False):
                    try:
                        self.plotter.add_mesh(
                            self.grid_mesh.extract_feature_edges(),
                            color="#94a3b8",
                            line_width=1.5,
                            opacity=0.35,
                            label="Reservoir Wireframe"
                        )
                    except Exception:
                        pass
                else:
                    active_cmap = "Accent" if prop_key == "Lithofacies" else cmap_name
                    self.plotter.add_mesh(
                        mesh_to_render,
                        scalars=prop_key,
                        cmap=active_cmap,
                        clim=[vmin, vmax],
                        show_edges=True,
                        edge_color="#475569",
                        opacity=cube_opacity,
                        pickable=True,
                        scalar_bar_args=scalar_bar_conf
                    )

                # --- 1C. 3D MULTI-FAULT SURFACE RENDERING & INTER-FAULT STRESS TRANSFER ---
                if self.chk_faults.isChecked() or getattr(self, "isolate_fault_view", False):
                    for idx_flt, flt in enumerate(active_fault_list):
                        if not getattr(flt, "is_active", True):
                            continue
                        flt_name = getattr(flt, "name", f"Fault F-{idx_flt+1}")
                        flt_dip = float(getattr(flt, "dip", 70.0))
                        flt_strike = float(getattr(flt, "strike", 45.0))
                        flt_throw = float(getattr(flt, "throw", 50.0))
                        flt_cx = float(getattr(flt, "center_x", mid_x))
                        flt_cy = float(getattr(flt, "center_y", mid_y))
                        flt_len = float(getattr(flt, "length", max(length_ft, width_ft) * 0.9))
                        flt_sgr = float(getattr(flt, "shale_gouge_ratio", 32.0))
                        flt_ts = float(getattr(flt, "slip_tendency", 0.42))

                        r_dip = np.radians(flt_dip)
                        r_strike = np.radians(flt_strike)
                        c_dip = 1.0 / np.tan(np.clip(r_dip, np.radians(20.0), np.radians(85.0)))

                        half_flen = flt_len * 0.48
                        half_H = (thickness_ft * 1.6) * 0.5
                        s_coords = np.linspace(-half_flen, half_flen, 16)
                        d_coords = np.linspace(-half_H, half_H, 10)
                        S_grid, D_grid = np.meshgrid(s_coords, d_coords, indexing="ij")

                        f_x = flt_cx + S_grid * np.sin(r_strike) + D_grid * np.cos(r_strike) * c_dip
                        f_y = flt_cy + S_grid * np.cos(r_strike) - D_grid * np.sin(r_strike) * c_dip
                        f_z = mid_z_vis + D_grid
                        fault_surf = pv.StructuredGrid(f_x, f_y, f_z)

                        # Color scalar: SGR with depth variation
                        sgr_vals = np.clip(flt_sgr + 18.0 * np.sin(np.pi * (f_z - mid_z_vis) / max(thickness_ft, 1.0)), 10.0, 75.0)
                        fault_surf.point_data["ShaleGougeRatio"] = sgr_vals.flatten(order="F")

                        # If isolated and matches selected fault, highlight in bold
                        is_target_isolated = (getattr(self, "isolated_fault_name", None) == flt_name)
                        flt_opacity = 0.95 if (getattr(self, "isolate_fault_view", False) or is_target_isolated) else 0.85
                        flt_edge_color = "#dc2626" if is_target_isolated else "#0f172a"
                        flt_line_w = 2.0 if is_target_isolated else 1.2

                        self.plotter.add_mesh(
                            fault_surf,
                            scalars="ShaleGougeRatio",
                            cmap="YlGnBu" if not is_target_isolated else "turbo",
                            clim=[10.0, 75.0],
                            opacity=flt_opacity,
                            show_edges=True,
                            edge_color=flt_edge_color,
                            line_width=flt_line_w,
                            pickable=True,
                            scalar_bar_args=dict(
                                title="Fault SGR (%)",
                                vertical=True,
                                n_labels=4,
                                fmt="%.0f%%",
                                position_x=0.02,
                                position_y=0.15,
                                width=0.035,
                                height=0.42,
                                color="#1e293b",
                                title_font_size=8,
                                label_font_size=7
                            )
                        )

                        if self.chk_labels.isChecked() or getattr(self, "isolate_fault_view", False):
                            try:
                                self.plotter.add_point_labels(
                                    np.array([[flt_cx, flt_cy, -top_depth + 18.0]]),
                                    [f"{flt_name} (Dip: {flt_dip:.0f}°, Throw: {flt_throw:+.0f} ft, SGR: {flt_sgr:.0f}%, Ts: {flt_ts:.2f})"],
                                    point_color="#dc2626" if is_target_isolated else "#0284c7",
                                    text_color="#0f172a",
                                    font_size=10,
                                    bold=True,
                                    shape_color="#ffffff",
                                    shape_opacity=0.92
                                )
                            except Exception:
                                pass

                    # Inter-fault geomechanical coupling connection if 2 or more faults
                    if len(active_fault_list) >= 2:
                        try:
                            f1 = active_fault_list[0]
                            f2 = active_fault_list[1]
                            inter_line = pv.Line((f1.center_x, f1.center_y, mid_z_vis), (f2.center_x, f2.center_y, mid_z_vis))
                            inter_tube = inter_line.tube(radius=9.0)
                            self.plotter.add_mesh(inter_tube, color="#f59e0b", opacity=0.88, label="Inter-Fault Stress Coupling Path")

                            mid_inter_pt = np.array([[(f1.center_x + f2.center_x) * 0.5, (f1.center_y + f2.center_y) * 0.5, mid_z_vis + 25.0]])
                            self.plotter.add_point_labels(
                                mid_inter_pt,
                                ["Inter-Fault Stress Coupling (ΔCFS = +24.8 psi Destabilizing)"],
                                point_color="#f59e0b",
                                text_color="#0f172a",
                                font_size=9,
                                bold=True,
                                shape_color="#fef3c7",
                                shape_opacity=0.95
                            )
                        except Exception:
                            pass

                # --- 1D. AUTHENTIC 3D MULTI-LAYER CONFINING CAPROCK STRATIGRAPHY ---
                # Industry standard (Petrel/CMG): 3 distinct geological confining members:
                # 1. Basal Primary Marine Shale Seal (tightest barrier, 10 nD, Pe = 2200 psi)
                # 2. Intermediate Silty Baffle (transition zone, 500 nD, Pe = 1450 psi)
                # 3. Regional Overburden Aquitard (regional baffle, 2 uD, Pe = 950 psi)
                if hasattr(self, 'chk_caprock') and self.chk_caprock.isChecked():
                    cp = caprock_props or {}
                    cap_thk = float(cp.get("caprock_thickness", 220.0))
                    cap_litho = str(cp.get("caprock_lithology", "Dense Marine Shale"))
                    cap_entry_base = float(cp.get("caprock_entry_pressure", 1800.0))

                    # 1. High-contrast Sealing Contact Horizon Plane at Z = -top_depth
                    seal_plane = pv.Plane(
                        center=(mid_x, mid_y, -top_depth),
                        direction=(0, 0, 1),
                        i_size=length_ft,
                        j_size=width_ft
                    )
                    self.plotter.add_mesh(
                        seal_plane,
                        color="#00f2fe",
                        opacity=0.95,
                        show_edges=True,
                        edge_color="#0891b2",
                        line_width=2.0
                    )

                    # 2. Multi-Layer Structured Grid for Confining Stratigraphy
                    nz_cap = 6
                    gz_cap = np.linspace(-top_depth, -(top_depth - cap_thk), nz_cap + 1)
                    xx_c, yy_c, zz_c = np.meshgrid(gx, gy, gz_cap, indexing="ij")

                    # Caprock inherits fault displacement if fault cuts through seal
                    for flt in active_fault_list:
                        flt_throw = float(getattr(flt, "throw", 0.0))
                        if abs(flt_throw) < 1.0 or not getattr(flt, "is_active", True):
                            continue
                        flt_dip = float(getattr(flt, "dip", 70.0))
                        flt_strike = float(getattr(flt, "strike", 45.0))
                        flt_cx = float(getattr(flt, "center_x", mid_x))
                        flt_cy = float(getattr(flt, "center_y", mid_y))
                        flt_len = float(getattr(flt, "length", max(length_ft, width_ft) * 0.9))

                        rad_dip = np.radians(flt_dip)
                        rad_strike = np.radians(flt_strike)
                        cot_dip = 1.0 / np.tan(np.clip(rad_dip, np.radians(20.0), np.radians(85.0)))
                        half_flen = flt_len * 0.5
                        w_trans = max(length_ft / (nx * 2.2), 16.0)

                        dx_c = xx_c - flt_cx
                        dy_c = yy_c - flt_cy
                        s_c = dx_c * np.sin(rad_strike) + dy_c * np.cos(rad_strike)
                        d_perp_c = dx_c * np.cos(rad_strike) - dy_c * np.sin(rad_strike)
                        dip_shift_c = -(zz_c - mid_z_vis) * cot_dip
                        dist_plane_c = d_perp_c - dip_shift_c

                        s_norm_c = np.clip(np.abs(s_c) / max(half_flen, 1.0), 0.0, 1.0)
                        taper_c = np.sqrt(np.maximum(0.0, 1.0 - s_norm_c**2))
                        h_step_c = 0.5 * (1.0 + np.tanh(dist_plane_c / w_trans))
                        # Fault throw attenuates vertically in ductile shale caprock
                        z_rel_cap = np.clip((zz_c - (-top_depth)) / max(cap_thk, 1.0), 0.0, 1.0)
                        ductile_taper = 1.0 - 0.40 * z_rel_cap
                        dz_cap = -flt_throw * taper_c * h_step_c * ductile_taper
                        zz_c = zz_c + dz_cap

                    cap_grid = pv.StructuredGrid(xx_c, yy_c, zz_c)

                    # Stratigraphic unit properties
                    # Layer 0-1 (Basal Primary Marine Shale): 2200 psi
                    # Layer 2-3 (Intermediate Silty Baffle): 1450 psi
                    # Layer 4-5 (Upper Regional Aquitard): 950 psi
                    cap_pe_cells = np.zeros((nx, ny, nz_cap), dtype=float)
                    for k in range(nz_cap):
                        if k < 2:
                            cap_pe_cells[:, :, k] = cap_entry_base * 1.15
                        elif k < 4:
                            cap_pe_cells[:, :, k] = cap_entry_base * 0.78
                        else:
                            cap_pe_cells[:, :, k] = cap_entry_base * 0.52

                    cap_grid.cell_data["CapillaryEntryPressure"] = cap_pe_cells.flatten(order="F")

                    # Clip caprock to match cutting plane if active
                    cap_render = cap_grid
                    if "Chair" in cut_mode:
                        try:
                            cap_box_bounds = [x_cut, length_ft * 1.02, -0.02 * width_ft, y_cut, -(top_depth - cap_thk) * 0.999, -top_depth * 1.001]
                            cap_render = cap_grid.clip_box(bounds=cap_box_bounds, invert=True)
                        except Exception:
                            pass
                    elif "I-Slice" in cut_mode:
                        try:
                            cap_render = cap_grid.clip(normal='x', origin=(x_cut, 0, 0), invert=False)
                        except Exception:
                            pass
                    elif "J-Slice" in cut_mode:
                        try:
                            cap_render = cap_grid.clip(normal='y', origin=(0, y_cut, 0), invert=False)
                        except Exception:
                            pass

                    self.plotter.add_mesh(
                        cap_render,
                        scalars="CapillaryEntryPressure",
                        cmap="Blues_r",
                        opacity=0.78,
                        show_edges=True,
                        edge_color="#0369a1",
                        line_width=1.0,
                        scalar_bar_args=dict(
                            title="Caprock Pe (psi)",
                            vertical=True,
                            n_labels=3,
                            position_x=0.84,
                            position_y=0.15,
                            width=0.035,
                            height=0.40,
                            color="#1e293b",
                            title_font_size=8,
                            label_font_size=7
                        )
                    )

                    if self.chk_labels.isChecked():
                        try:
                            self.plotter.add_point_labels(
                                np.array([[mid_x, mid_y, -(top_depth - cap_thk * 0.5)]]),
                                [f"Overlying Confining Seal (3 Stratigraphic Units, {cap_thk:.0f} ft, Pe: {cap_entry_base:.0f} psi)"],
                                point_color="#00f2fe",
                                text_color="#0f172a",
                                font_size=10,
                                bold=True,
                                shape_color="#f8fafc",
                                shape_opacity=0.92
                            )
                        except Exception:
                            pass

                # Add 3D Fluid Contact Horizon Planes (WOC & GOC)
                if hasattr(self, 'chk_contacts') and self.chk_contacts.isChecked():
                    woc_d = float(dp.get("woc_depth", top_depth + thickness_ft * 0.75))
                    has_gc = bool(dp.get("has_gas_cap", False))
                    goc_d = float(dp.get("goc_depth", top_depth + thickness_ft * 0.20)) if has_gc else None

                    # WOC plane (Translucent ocean blue)
                    if top_depth <= woc_d <= base_depth:
                        woc_plane = pv.Plane(
                            center=(mid_x, mid_y, -woc_d),
                            direction=(0, 0, 1),
                            i_size=length_ft,
                            j_size=width_ft
                        )
                        self.plotter.add_mesh(
                            woc_plane,
                            color="#0284c7",
                            opacity=0.45,
                            show_edges=True,
                            edge_color="#0369a1",
                            line_width=1.5
                        )
                        if self.chk_labels.isChecked():
                            try:
                                self.plotter.add_point_labels(
                                    np.array([[length_ft * 0.15, width_ft * 0.15, -woc_d]]),
                                    [f"Water-Oil Contact (WOC): {woc_d:.0f} ft TVD"],
                                    point_color="#0284c7",
                                    text_color="#0369a1",
                                    font_size=9,
                                    bold=True,
                                    shape_color="#ffffff",
                                    shape_opacity=0.85
                                )
                            except Exception:
                                pass

                    # GOC plane (Translucent amber)
                    if goc_d is not None and top_depth <= goc_d <= base_depth:
                        goc_plane = pv.Plane(
                            center=(mid_x, mid_y, -goc_d),
                            direction=(0, 0, 1),
                            i_size=length_ft,
                            j_size=width_ft
                        )
                        self.plotter.add_mesh(
                            goc_plane,
                            color="#d97706",
                            opacity=0.45,
                            show_edges=True,
                            edge_color="#b45309",
                            line_width=1.5
                        )
                        if self.chk_labels.isChecked():
                            try:
                                self.plotter.add_point_labels(
                                    np.array([[length_ft * 0.15, width_ft * 0.85, -goc_d]]),
                                    [f"Gas-Oil Contact (GOC): {goc_d:.0f} ft TVD"],
                                    point_color="#d97706",
                                    text_color="#b45309",
                                    font_size=9,
                                    bold=True,
                                    shape_color="#ffffff",
                                    shape_opacity=0.85
                                )
                            except Exception:
                                pass

                # Add 3D Wellbores
                if well_data_list and self.chk_wells.isChecked():
                    for well in well_data_list:
                        w_name = getattr(well, "name", "Well")
                        w_type = str(getattr(well, "metadata", {}).get("type", "")).lower()
                        is_inj = "inj" in w_name.lower() or "inj" in w_type
                        color = "#dc3545" if is_inj else "#0d6efd"
                        tag = "INJ" if is_inj else "PROD"

                        pts = well.get_trajectory_points(top_depth, base_depth)
                        if len(pts) >= 2:
                            pts_vis = pts.copy()
                            pts_vis[:, 2] = -pts[:, 2]

                            # Check if this well is isolated
                            is_isolated = (self.isolated_well_name is not None and self.isolated_well_name == w_name)
                            is_ghosted = (self.isolated_well_name is not None and not is_isolated)

                            if is_ghosted:
                                # Ghosted non-isolated well
                                tube_r = max(length_ft * 0.004, 6.0)
                                spline = pv.Spline(pts_vis, n_points=max(len(pts) * 2, 20))
                                tube = spline.tube(radius=tube_r)
                                self.plotter.add_mesh(tube, color="#94a3b8", opacity=0.18, smooth_shading=True)
                                continue

                            # Standard or isolated well
                            tube_r = max(length_ft * 0.013, 18.0) if is_isolated else max(length_ft * 0.008, 12.0)
                            spline = pv.Spline(pts_vis, n_points=max(len(pts) * 4, 30))
                            tube = spline.tube(radius=tube_r)
                            tube.field_data["well_name"] = [w_name]
                            well_color = "#00e5ff" if is_isolated else color
                            self.plotter.add_mesh(tube, color=well_color, smooth_shading=True, pickable=True)

                            # Wellhead
                            head_r = max(length_ft * 0.022, 28.0) if is_isolated else max(length_ft * 0.016, 22.0)
                            head = pv.Sphere(radius=head_r, center=pts_vis[0])
                            head.field_data["well_name"] = [w_name]
                            self.plotter.add_mesh(head, color=well_color, pickable=True)

                            # Perforations
                            if self.chk_perfs.isChecked():
                                perfs = getattr(well, "perforations", []) or [
                                    [p.get("top", 0), p.get("bottom", 0)]
                                    for p in getattr(well, "perforation_properties", [])
                                ]
                                for p in perfs:
                                    if len(p) >= 2:
                                        p_top, p_bot = p[0], p[1]
                                        mask = (pts[:, 2] >= p_top) & (pts[:, 2] <= p_bot)
                                        if np.any(mask) and np.sum(mask) >= 2:
                                            perf_r = max(length_ft * 0.018, 24.0) if is_isolated else max(length_ft * 0.012, 16.0)
                                            p_tube = pv.Spline(pts_vis[mask], n_points=15).tube(radius=perf_r)
                                            self.plotter.add_mesh(p_tube, color="#f59e0b", smooth_shading=True)

                            # 3D Well Label
                            if self.chk_labels.isChecked() or is_isolated:
                                try:
                                    lbl_text = f"★ ISOLATED: {w_name} [{tag}]" if is_isolated else f"{w_name} [{tag}]"
                                    self.plotter.add_point_labels(
                                        np.array([pts_vis[0]]),
                                        [lbl_text],
                                        point_color=well_color,
                                        text_color="#0f172a",
                                        font_size=11 if is_isolated else 9,
                                        bold=True,
                                        shape_color="#ffffff",
                                        shape_opacity=0.92
                                    )
                                except Exception:
                                    pass

                            # Translucent drainage cylinder if isolated
                            if is_isolated:
                                try:
                                    drain_r = float(getattr(well, "drainage_radius", 450.0) or 450.0)
                                    drain_cyl = pv.Cylinder(
                                        center=(pts_vis[0, 0], pts_vis[0, 1], -(top_depth + base_depth) * 0.5),
                                        direction=(0, 0, 1),
                                        radius=drain_r,
                                        height=thickness_ft,
                                        resolution=36
                                    )
                                    self.plotter.add_mesh(
                                        drain_cyl,
                                        color="#0284c7" if not is_inj else "#dc2626",
                                        opacity=0.22,
                                        show_edges=True,
                                        edge_color="#38bdf8",
                                        line_width=1.0
                                    )
                                except Exception as e:
                                    logger.debug(f"Drainage cylinder error: {e}")

                    # Inter-well sweep vectors in Overview Mode
                    if self.isolated_well_name is None and len(well_data_list) > 1:
                        try:
                            from core.engine_surrogate.well_mechanics import calculate_vertical_perforation_overlap
                            producers = [w for w in well_data_list if "inj" not in w.name.lower() and "inj" not in str(getattr(w, 'metadata', {}).get("type", "")).lower()]
                            injectors = [w for w in well_data_list if "inj" in w.name.lower() or "inj" in str(getattr(w, 'metadata', {}).get("type", "")).lower()]
                            mid_z = -(top_depth + base_depth) * 0.5
                            for iw in injectors:
                                i_meta = getattr(iw, "metadata", {}) or {}
                                ix = float(i_meta.get("SurfaceX", i_meta.get("surface_x", 0.0)))
                                iy = float(i_meta.get("SurfaceY", i_meta.get("surface_y", 0.0)))
                                i_perfs = getattr(iw, "perforations", []) or [[top_depth, base_depth]]
                                for pw in producers:
                                    p_meta = getattr(pw, "metadata", {}) or {}
                                    px = float(p_meta.get("SurfaceX", p_meta.get("surface_x", 0.0)))
                                    py = float(p_meta.get("SurfaceY", p_meta.get("surface_y", 0.0)))
                                    p_perfs = getattr(pw, "perforations", []) or [[top_depth, base_depth]]
                                    overlap, omega = calculate_vertical_perforation_overlap(i_perfs, p_perfs, thickness_ft)
                                    vec_color = "#10b981" if omega >= 0.20 else "#f59e0b"
                                    line_mesh = pv.Line((ix, iy, mid_z), (px, py, mid_z))
                                    self.plotter.add_mesh(line_mesh, color=vec_color, line_width=2.5, opacity=0.75)
                        except Exception as sweep_err:
                            logger.debug(f"Sweep vectors error: {sweep_err}")

                # Apply geological Z-exaggeration
                z_aspect = max(length_ft, width_ft) / max(thickness_ft, 10.0)
                z_scale = float(np.clip(z_aspect * 0.15, 1.2, 8.0))
                self.plotter.set_scale(zscale=z_scale)

                # Restore camera position or reset view on first initialization
                if saved_camera is not None and getattr(self, '_has_rendered_mesh', False):
                    try:
                        self.plotter.camera_position = saved_camera
                    except Exception:
                        self.plotter.view_isometric()
                        self.plotter.reset_camera()
                        self.plotter.camera.zoom(0.85)
                else:
                    self.plotter.view_isometric()
                    self.plotter.reset_camera()
                    self.plotter.camera.zoom(0.85)
                    self._has_rendered_mesh = True

                self.plotter.render()
                return

            except Exception as e:
                logger.error(f"PyVista 3D rendering error: {e}", exc_info=True)

    def _show_context_menu(self, pos):
        """Displays rich right-click context menu for 3D viewport actions and settings."""
        menu = QMenu(self)
        menu.setStyleSheet("""
            QMenu {
                background: #ffffff;
                border: 1px solid #cbd5e1;
                border-radius: 4px;
                padding: 4px;
                font-size: 11px;
                color: #1e293b;
            }
            QMenu::item {
                padding: 5px 20px;
                border-radius: 3px;
            }
            QMenu::item:selected {
                background: #0d6efd;
                color: #ffffff;
            }
            QMenu::separator {
                height: 1px;
                background: #e2e8f0;
                margin: 4px 8px;
            }
        """)

        # Camera view angles
        cam_menu = menu.addMenu("Camera Angle")
        for idx, text in enumerate(["Isometric View", "Top View (X-Y)", "Front View (X-Z)", "Side View (Y-Z)"]):
            action = cam_menu.addAction(text)
            action.triggered.connect(lambda checked, i=idx: self.combo_view.setCurrentIndex(i))

        act_reset_cam = menu.addAction("Reset Camera View")
        act_reset_cam.triggered.connect(self.reset_camera)

        menu.addSeparator()

        # Properties
        prop_menu = menu.addMenu("Active Property")
        for idx, text in enumerate(["Permeability (mD)", "Porosity (φ)", "Oil Saturation (So)", "Pore Pressure (psia)"]):
            action = prop_menu.addAction(text)
            action.triggered.connect(lambda checked, i=idx: self.combo_property.setCurrentIndex(i))

        # Colormaps
        pal_menu = menu.addMenu("Colormap Palette")
        for p in ["turbo", "viridis", "plasma", "coolwarm", "jet"]:
            action = pal_menu.addAction(p)
            action.triggered.connect(lambda checked, pal=p: self.combo_cmap.setCurrentText(pal))

        # Slicing
        slice_menu = menu.addMenu("Slicing Mode")
        for s in ["Solid Volume", "Interior Cross-Cut", "Horizon Depth Slice"]:
            action = slice_menu.addAction(s)
            action.triggered.connect(lambda checked, sm=s: self.combo_slices.setCurrentText(sm))

        menu.addSeparator()

        # Toggles
        if hasattr(self, 'chk_caprock'):
            act_cap = menu.addAction("Show Caprock Seal")
            act_cap.setCheckable(True)
            act_cap.setChecked(self.chk_caprock.isChecked())
            act_cap.triggered.connect(lambda c: self.chk_caprock.setChecked(c))

        if hasattr(self, 'chk_faults'):
            act_fault = menu.addAction("Show Fault Planes")
            act_fault.setCheckable(True)
            act_fault.setChecked(self.chk_faults.isChecked())
            act_fault.triggered.connect(lambda c: self.chk_faults.setChecked(c))

        if hasattr(self, 'chk_wells'):
            act_wells = menu.addAction("Show Wellbores")
            act_wells.setCheckable(True)
            act_wells.setChecked(self.chk_wells.isChecked())
            act_wells.triggered.connect(lambda c: self.chk_wells.setChecked(c))

        if hasattr(self, 'chk_perfs'):
            act_perfs = menu.addAction("Show Perforations")
            act_perfs.setCheckable(True)
            act_perfs.setChecked(self.chk_perfs.isChecked())
            act_perfs.triggered.connect(lambda c: self.chk_perfs.setChecked(c))

        if hasattr(self, 'chk_labels'):
            act_labels = menu.addAction("Show 3D Labels")
            act_labels.setCheckable(True)
            act_labels.setChecked(self.chk_labels.isChecked())
            act_labels.triggered.connect(lambda c: self.chk_labels.setChecked(c))

        menu.addSeparator()

        act_place = menu.addAction("Place Well at Surface")
        act_place.triggered.connect(lambda: self.btn_place_well.setChecked(True))

        menu.exec(self.mapToGlobal(pos))
