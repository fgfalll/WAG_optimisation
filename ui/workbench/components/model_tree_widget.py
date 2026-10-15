import logging
from typing import Optional, List
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QTreeWidget, QTreeWidgetItem, QPushButton,
    QLabel, QHBoxLayout, QFrame, QMenu
)
from PyQt6.QtGui import QIcon, QFont, QColor
from PyQt6.QtCore import pyqtSignal, Qt, QSize

from .subsurface_icons import create_subsurface_icon

logger = logging.getLogger(__name__)


class ModelTreeWidget(QFrame):
    """
    Hierarchical project tree acting as the central mental map of the reservoir model.
    Provides clear empty-state call-to-actions, rich domain icons, and context selection.
    """
    node_selected = pyqtSignal(str, str) # (domain, item_key)
    add_well_requested = pyqtSignal()
    add_layer_requested = pyqtSignal()
    generate_pattern_requested = pyqtSignal(str) # "5spot", "9spot", "linedrive"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setStyleSheet("""
            QFrame {
                background: #ffffff;
                border: 1px solid #dee2e6;
                border-radius: 4px;
            }
            QTreeWidget {
                background: #ffffff;
                border: none;
                color: #212529;
                font-size: 11px;
            }
            QTreeWidget::item {
                padding: 4px 6px;
                border-radius: 3px;
                color: #212529;
            }
            QTreeWidget::item:hover {
                background: #f1f5f9;
            }
            QTreeWidget::item:selected {
                background: #e7f1ff;
                color: #0d6efd;
                font-weight: bold;
            }
            QLabel#treeHeader {
                color: #1e293b;
                font-weight: bold;
                font-size: 11px;
                padding: 6px 8px;
                background: #f8fafc;
                border-bottom: 1px solid #dee2e6;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(4)

        header = QLabel("Model Tree")
        header.setObjectName("treeHeader")
        layout.addWidget(header)

        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.setIconSize(QSize(18, 18))
        self.tree.setIndentation(16)
        self.tree.setUniformRowHeights(True)
        self.tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self._show_tree_context_menu)
        self.tree.itemClicked.connect(self._on_item_clicked)
        self.tree.currentItemChanged.connect(self._on_current_item_changed)
        layout.addWidget(self.tree, stretch=1)

        # Quick action footer
        footer_frame = QFrame()
        footer_frame.setStyleSheet("background: #f8fafc; border-top: 1px solid #dee2e6;")
        footer_layout = QHBoxLayout(footer_frame)
        footer_layout.setContentsMargins(4, 4, 4, 4)
        footer_layout.setSpacing(4)

        btn_add_well = QPushButton("Add Well")
        btn_add_well.setIcon(create_subsurface_icon("wells", 14))
        btn_add_well.setToolTip("Add a new well to the network")
        btn_add_well.setStyleSheet("background: #ffffff; color: #0d6efd; font-weight: bold; font-size: 10px; padding: 3px 6px; border: 1px solid #ced4da; border-radius: 3px;")
        btn_add_well.clicked.connect(self.add_well_requested.emit)
        footer_layout.addWidget(btn_add_well)

        btn_add_layer = QPushButton("Add Layer")
        btn_add_layer.setIcon(create_subsurface_icon("layers", 14))
        btn_add_layer.setToolTip("Add a geological layer")
        btn_add_layer.setStyleSheet("background: #ffffff; color: #198754; font-weight: bold; font-size: 10px; padding: 3px 6px; border: 1px solid #ced4da; border-radius: 3px;")
        btn_add_layer.clicked.connect(self.add_layer_requested.emit)
        footer_layout.addWidget(btn_add_layer)

        btn_pattern = QPushButton("5-Spot")
        btn_pattern.setIcon(create_subsurface_icon("reservoir", 14))
        btn_pattern.setToolTip("Generate standard 5-spot pattern")
        btn_pattern.setStyleSheet("background: #ffffff; color: #495057; font-weight: bold; font-size: 10px; padding: 3px 6px; border: 1px solid #ced4da; border-radius: 3px;")
        btn_pattern.clicked.connect(lambda: self.generate_pattern_requested.emit("5spot"))
        footer_layout.addWidget(btn_pattern)

        layout.addWidget(footer_frame)

        self._build_initial_tree()

    def _build_initial_tree(self):
        self.tree.clear()

        # 1. Reservoir Geometry
        self.res_root = QTreeWidgetItem(["Reservoir Framework"])
        self.res_root.setIcon(0, create_subsurface_icon("reservoir"))
        self.res_root.setData(0, Qt.ItemDataRole.UserRole, ("reservoir", "root"))
        
        self.item_3d = QTreeWidgetItem(["3D Volumetric Reservoir Studio"])
        self.item_3d.setIcon(0, create_subsurface_icon("reservoir"))
        self.item_3d.setData(0, Qt.ItemDataRole.UserRole, ("reservoir", "studio_3d"))
        self.res_root.addChild(self.item_3d)

        # Parent item: Grid Dimensions & Mesh
        self.item_grid = QTreeWidgetItem(["Grid Dimensions (NX, NY, NZ)"])
        self.item_grid.setIcon(0, create_subsurface_icon("grid"))
        self.item_grid.setData(0, Qt.ItemDataRole.UserRole, ("reservoir", "grid"))
        
        item_grid_tbl = QTreeWidgetItem(["Grid Geometry & Coordinates (Table)"])
        item_grid_tbl.setIcon(0, create_subsurface_icon("table"))
        item_grid_tbl.setData(0, Qt.ItemDataRole.UserRole, ("reservoir", "grid_table"))
        self.item_grid.addChild(item_grid_tbl)

        item_grid_grp = QTreeWidgetItem(["Layer Depth & Thickness Profile (Graph)"])
        item_grid_grp.setIcon(0, create_subsurface_icon("graph"))
        item_grid_grp.setData(0, Qt.ItemDataRole.UserRole, ("reservoir", "grid_graph"))
        self.item_grid.addChild(item_grid_grp)
        self.res_root.addChild(self.item_grid)

        # Parent item: Volumetrics & OOIP
        self.item_ooip = QTreeWidgetItem(["Volumetrics & OOIP / Storage"])
        self.item_ooip.setIcon(0, create_subsurface_icon("volumetrics"))
        self.item_ooip.setData(0, Qt.ItemDataRole.UserRole, ("reservoir", "ooip"))

        item_ooip_tbl = QTreeWidgetItem(["Volumetric OOIP Material Balance (Table)"])
        item_ooip_tbl.setIcon(0, create_subsurface_icon("table"))
        item_ooip_tbl.setData(0, Qt.ItemDataRole.UserRole, ("reservoir", "ooip_table"))
        self.item_ooip.addChild(item_ooip_tbl)

        item_ooip_grp = QTreeWidgetItem(["OOIP & CO2 Storage Waterfall (Graph)"])
        item_ooip_grp.setIcon(0, create_subsurface_icon("graph"))
        item_ooip_grp.setData(0, Qt.ItemDataRole.UserRole, ("reservoir", "ooip_graph"))
        self.item_ooip.addChild(item_ooip_grp)
        self.res_root.addChild(self.item_ooip)

        # Parent item: Stratigraphy & Geology
        self.item_strat = QTreeWidgetItem(["Stratigraphy & Cross-Section"])
        self.item_strat.setIcon(0, create_subsurface_icon("stratigraphy"))
        self.item_strat.setData(0, Qt.ItemDataRole.UserRole, ("reservoir", "stratigraphy"))

        item_strat_grp = QTreeWidgetItem(["2D Stratigraphic Cross-Section (Graph)"])
        item_strat_grp.setIcon(0, create_subsurface_icon("graph"))
        item_strat_grp.setData(0, Qt.ItemDataRole.UserRole, ("reservoir", "stratigraphy_graph"))
        self.item_strat.addChild(item_strat_grp)

        item_strat_tbl = QTreeWidgetItem(["Zonal Stratigraphy & Layer Tops (Table)"])
        item_strat_tbl.setIcon(0, create_subsurface_icon("table"))
        item_strat_tbl.setData(0, Qt.ItemDataRole.UserRole, ("reservoir", "stratigraphy_table"))
        self.item_strat.addChild(item_strat_tbl)
        self.res_root.addChild(self.item_strat)

        self.tree.addTopLevelItem(self.res_root)

        # 2. Petrophysics & Heterogeneity
        self.petro_root = QTreeWidgetItem(["Petrophysics & Rock"])
        self.petro_root.setIcon(0, create_subsurface_icon("rock"))
        self.petro_root.setData(0, Qt.ItemDataRole.UserRole, ("petrophysics", "root"))

        # Parent item: Rock Properties & Realization
        self.item_rock = QTreeWidgetItem(["Rock Properties & Realization"])
        self.item_rock.setIcon(0, create_subsurface_icon("rock"))
        self.item_rock.setData(0, Qt.ItemDataRole.UserRole, ("petrophysics", "rock"))

        item_rock_tbl = QTreeWidgetItem(["Petrophysical Realization Summary (Table)"])
        item_rock_tbl.setIcon(0, create_subsurface_icon("table"))
        item_rock_tbl.setData(0, Qt.ItemDataRole.UserRole, ("petrophysics", "rock_table"))
        self.item_rock.addChild(item_rock_tbl)

        item_rock_grp = QTreeWidgetItem(["Porosity vs Permeability Cross-Plot (Graph)"])
        item_rock_grp.setIcon(0, create_subsurface_icon("graph"))
        item_rock_grp.setData(0, Qt.ItemDataRole.UserRole, ("petrophysics", "rock_graph"))
        self.item_rock.addChild(item_rock_grp)
        self.petro_root.addChild(self.item_rock)

        # Parent item: Corey Relative Permeability
        self.item_relperm = QTreeWidgetItem(["Corey Relative Permeability"])
        self.item_relperm.setIcon(0, create_subsurface_icon("relperm"))
        self.item_relperm.setData(0, Qt.ItemDataRole.UserRole, ("petrophysics", "relperm"))

        item_relperm_grp = QTreeWidgetItem(["Corey 3-Phase Kr Curves (Graph)"])
        item_relperm_grp.setIcon(0, create_subsurface_icon("graph"))
        item_relperm_grp.setData(0, Qt.ItemDataRole.UserRole, ("petrophysics", "relperm_graph"))
        self.item_relperm.addChild(item_relperm_grp)

        item_relperm_tbl = QTreeWidgetItem(["Relative Permeability Endpoints (Table)"])
        item_relperm_tbl.setIcon(0, create_subsurface_icon("table"))
        item_relperm_tbl.setData(0, Qt.ItemDataRole.UserRole, ("petrophysics", "relperm_table"))
        self.item_relperm.addChild(item_relperm_tbl)
        self.petro_root.addChild(self.item_relperm)

        # Parent item: Geostatistics & Spatial Continuity
        self.item_geostat = QTreeWidgetItem(["Geostatistics & Variogram"])
        self.item_geostat.setIcon(0, create_subsurface_icon("geostat"))
        self.item_geostat.setData(0, Qt.ItemDataRole.UserRole, ("petrophysics", "geostat"))

        item_geostat_grp = QTreeWidgetItem(["Semivariogram Spatial Continuity (Graph)"])
        item_geostat_grp.setIcon(0, create_subsurface_icon("graph"))
        item_geostat_grp.setData(0, Qt.ItemDataRole.UserRole, ("petrophysics", "geostat_graph"))
        self.item_geostat.addChild(item_geostat_grp)

        item_geostat_tbl = QTreeWidgetItem(["Variogram & Kriging Parameters (Table)"])
        item_geostat_tbl.setIcon(0, create_subsurface_icon("table"))
        item_geostat_tbl.setData(0, Qt.ItemDataRole.UserRole, ("petrophysics", "geostat_table"))
        self.item_geostat.addChild(item_geostat_tbl)
        self.petro_root.addChild(self.item_geostat)

        self.tree.addTopLevelItem(self.petro_root)

        # 3. Fluids & PVT
        self.pvt_root = QTreeWidgetItem(["Fluids & PVT Model"])
        self.pvt_root.setIcon(0, create_subsurface_icon("pvt"))
        self.pvt_root.setData(0, Qt.ItemDataRole.UserRole, ("pvt", "root"))

        # Parent item: Black Oil PVT
        self.item_pvt_props = QTreeWidgetItem(["Black Oil PVT Model"])
        self.item_pvt_props.setIcon(0, create_subsurface_icon("pvt"))
        self.item_pvt_props.setData(0, Qt.ItemDataRole.UserRole, ("pvt", "properties"))

        item_bo_grp = QTreeWidgetItem(["Black Oil Curves Bo, Rs, Viscosity (Graph)"])
        item_bo_grp.setIcon(0, create_subsurface_icon("graph"))
        item_bo_grp.setData(0, Qt.ItemDataRole.UserRole, ("pvt", "black_oil_graph"))
        self.item_pvt_props.addChild(item_bo_grp)

        item_bo_tbl = QTreeWidgetItem(["Black Oil Numerical PVT Table (Table)"])
        item_bo_tbl.setIcon(0, create_subsurface_icon("table"))
        item_bo_tbl.setData(0, Qt.ItemDataRole.UserRole, ("pvt", "black_oil_table"))
        self.item_pvt_props.addChild(item_bo_tbl)
        self.pvt_root.addChild(self.item_pvt_props)

        # Parent item: Solvent Swelling & Thinning
        self.item_swelling = QTreeWidgetItem(["Solvent Swelling & Viscosity Reduction"])
        self.item_swelling.setIcon(0, create_subsurface_icon("pvt"))
        self.item_swelling.setData(0, Qt.ItemDataRole.UserRole, ("pvt", "swelling"))

        item_sw_grp = QTreeWidgetItem(["CO2 Swelling & Viscosity Thinning (Graph)"])
        item_sw_grp.setIcon(0, create_subsurface_icon("graph"))
        item_sw_grp.setData(0, Qt.ItemDataRole.UserRole, ("pvt", "swelling_graph"))
        self.item_swelling.addChild(item_sw_grp)

        item_sw_tbl = QTreeWidgetItem(["Solvent Swelling & Dissolution Data (Table)"])
        item_sw_tbl.setIcon(0, create_subsurface_icon("table"))
        item_sw_tbl.setData(0, Qt.ItemDataRole.UserRole, ("pvt", "swelling_table"))
        self.item_swelling.addChild(item_sw_tbl)
        self.pvt_root.addChild(self.item_swelling)

        # Parent item: MMP
        self.item_mmp = QTreeWidgetItem(["Minimum Miscibility Pressure (MMP)"])
        self.item_mmp.setIcon(0, create_subsurface_icon("mmp"))
        self.item_mmp.setData(0, Qt.ItemDataRole.UserRole, ("pvt", "mmp"))

        item_mmp_grp = QTreeWidgetItem(["MMP Multi-Correlation Barometer (Graph)"])
        item_mmp_grp.setIcon(0, create_subsurface_icon("graph"))
        item_mmp_grp.setData(0, Qt.ItemDataRole.UserRole, ("pvt", "mmp_graph"))
        self.item_mmp.addChild(item_mmp_grp)

        item_mmp_tbl = QTreeWidgetItem(["MMP Correlations & Miscibility Margins (Table)"])
        item_mmp_tbl.setIcon(0, create_subsurface_icon("table"))
        item_mmp_tbl.setData(0, Qt.ItemDataRole.UserRole, ("pvt", "mmp_table"))
        self.item_mmp.addChild(item_mmp_tbl)
        self.pvt_root.addChild(self.item_mmp)

        # Parent item: Detailed Composition & EOS
        self.item_pvt_detailed = QTreeWidgetItem(["Compositional EOS & Phase Behavior"])
        self.item_pvt_detailed.setIcon(0, create_subsurface_icon("pvt"))
        self.item_pvt_detailed.setData(0, Qt.ItemDataRole.UserRole, ("pvt", "detailed"))

        item_eos_grp = QTreeWidgetItem(["P-T Phase Envelope & Critical Diagram (Graph)"])
        item_eos_grp.setIcon(0, create_subsurface_icon("graph"))
        item_eos_grp.setData(0, Qt.ItemDataRole.UserRole, ("pvt", "eos_graph"))
        self.item_pvt_detailed.addChild(item_eos_grp)

        item_eos_tbl = QTreeWidgetItem(["10-Component Detailed Composition (Table)"])
        item_eos_tbl.setIcon(0, create_subsurface_icon("table"))
        item_eos_tbl.setData(0, Qt.ItemDataRole.UserRole, ("pvt", "eos_table"))
        self.item_pvt_detailed.addChild(item_eos_tbl)
        self.pvt_root.addChild(self.item_pvt_detailed)

        # Parent item: Fluid Contacts & Columns
        self.item_contacts = QTreeWidgetItem(["Fluid Contacts & Hydrostatic Columns"])
        self.item_contacts.setIcon(0, create_subsurface_icon("stratigraphy"))
        self.item_contacts.setData(0, Qt.ItemDataRole.UserRole, ("pvt", "contacts"))

        item_fc_grp = QTreeWidgetItem(["Hydrostatic Pressure vs Depth Gradient (Graph)"])
        item_fc_grp.setIcon(0, create_subsurface_icon("graph"))
        item_fc_grp.setData(0, Qt.ItemDataRole.UserRole, ("pvt", "contacts_graph"))
        self.item_contacts.addChild(item_fc_grp)

        item_fc_tbl = QTreeWidgetItem(["Fluid Contacts & Zonal Depth Table (Table)"])
        item_fc_tbl.setIcon(0, create_subsurface_icon("table"))
        item_fc_tbl.setData(0, Qt.ItemDataRole.UserRole, ("pvt", "contacts_table"))
        self.item_contacts.addChild(item_fc_tbl)
        self.pvt_root.addChild(self.item_contacts)

        self.tree.addTopLevelItem(self.pvt_root)

        # 4. Well Network
        self.wells_root = QTreeWidgetItem(["Well Network"])
        self.wells_root.setIcon(0, create_subsurface_icon("wells"))
        self.wells_root.setData(0, Qt.ItemDataRole.UserRole, ("wells", "root"))

        item_winv = QTreeWidgetItem(["Master Well Inventory & Overview (Table)"])
        item_winv.setIcon(0, create_subsurface_icon("table"))
        item_winv.setData(0, Qt.ItemDataRole.UserRole, ("wells", "inventory_table"))
        self.wells_root.addChild(item_winv)

        item_wsched = QTreeWidgetItem(["Schedule & Operating Controls (Table)"])
        item_wsched.setIcon(0, create_subsurface_icon("schedule"))
        item_wsched.setData(0, Qt.ItemDataRole.UserRole, ("wells", "schedule_table"))
        self.wells_root.addChild(item_wsched)

        item_wgantt = QTreeWidgetItem(["Well Lifecycle & Development Timeline (Graph)"])
        item_wgantt.setIcon(0, create_subsurface_icon("gantt"))
        item_wgantt.setData(0, Qt.ItemDataRole.UserRole, ("wells", "schedule_gantt"))
        self.wells_root.addChild(item_wgantt)

        item_wwag = QTreeWidgetItem(["WAG Alternating Injection Schedule (Graph)"])
        item_wwag.setIcon(0, create_subsurface_icon("graph"))
        item_wwag.setData(0, Qt.ItemDataRole.UserRole, ("wells", "wag_schedule_graph"))
        self.wells_root.addChild(item_wwag)

        item_wsweep = QTreeWidgetItem(["Inter-Well Sweep & Connectivity (Graph)"])
        item_wsweep.setIcon(0, create_subsurface_icon("graph"))
        item_wsweep.setData(0, Qt.ItemDataRole.UserRole, ("wells", "sweep_matrix"))
        self.wells_root.addChild(item_wsweep)

        self.tree.addTopLevelItem(self.wells_root)

        # 5. Geomechanics & Containment
        self.geo_root = QTreeWidgetItem(["Geomechanics & Faults"])
        self.geo_root.setIcon(0, create_subsurface_icon("faults"))
        self.geo_root.setData(0, Qt.ItemDataRole.UserRole, ("geomechanics", "root"))

        # Parent item: Caprock Seal
        self.item_caprock = QTreeWidgetItem(["Caprock Seal & Containment"])
        self.item_caprock.setIcon(0, create_subsurface_icon("caprock"))
        self.item_caprock.setData(0, Qt.ItemDataRole.UserRole, ("geomechanics", "caprock"))

        item_cap_strat = QTreeWidgetItem(["Caprock Confining Stratigraphy & Thickness (Table)"])
        item_cap_strat.setIcon(0, create_subsurface_icon("table"))
        item_cap_strat.setData(0, Qt.ItemDataRole.UserRole, ("geomechanics", "caprock_strat_table"))
        self.item_caprock.addChild(item_cap_strat)

        item_cap_col = QTreeWidgetItem(["Capillary Sealing Column Height & Breakthrough (Graph)"])
        item_cap_col.setIcon(0, create_subsurface_icon("graph"))
        item_cap_col.setData(0, Qt.ItemDataRole.UserRole, ("geomechanics", "caprock_column_graph"))
        self.item_caprock.addChild(item_cap_col)

        item_cap_grp = QTreeWidgetItem(["Mohr-Coulomb Failure Envelopes (Graph)"])
        item_cap_grp.setIcon(0, create_subsurface_icon("graph"))
        item_cap_grp.setData(0, Qt.ItemDataRole.UserRole, ("geomechanics", "caprock_graph"))
        self.item_caprock.addChild(item_cap_grp)

        item_cap_tbl = QTreeWidgetItem(["Caprock Properties & Safety Factors (Table)"])
        item_cap_tbl.setIcon(0, create_subsurface_icon("table"))
        item_cap_tbl.setData(0, Qt.ItemDataRole.UserRole, ("geomechanics", "caprock_table"))
        self.item_caprock.addChild(item_cap_tbl)

        item_cap_3d = QTreeWidgetItem(["Caprock 3D Confining System (3D View)"])
        item_cap_3d.setIcon(0, create_subsurface_icon("caprock"))
        item_cap_3d.setData(0, Qt.ItemDataRole.UserRole, ("geomechanics", "caprock_3d"))
        self.item_caprock.addChild(item_cap_3d)
        self.geo_root.addChild(self.item_caprock)

        # Parent item: Fault Geometry & Slip
        self.item_faults = QTreeWidgetItem(["Fault Geometry & Slip Tendency"])
        self.item_faults.setIcon(0, create_subsurface_icon("faults"))
        self.item_faults.setData(0, Qt.ItemDataRole.UserRole, ("geomechanics", "faults"))

        item_flt_grp = QTreeWidgetItem(["3D Fault Plane Visualizer (3D View)"])
        item_flt_grp.setIcon(0, create_subsurface_icon("faults"))
        item_flt_grp.setData(0, Qt.ItemDataRole.UserRole, ("geomechanics", "fault_graph"))
        self.item_faults.addChild(item_flt_grp)

        item_flt_tbl = QTreeWidgetItem(["Fault Slip Tendency & SGR Seal (Table)"])
        item_flt_tbl.setIcon(0, create_subsurface_icon("table"))
        item_flt_tbl.setData(0, Qt.ItemDataRole.UserRole, ("geomechanics", "fault_table"))
        self.item_faults.addChild(item_flt_tbl)
        self.geo_root.addChild(self.item_faults)

        # Parent item: In-Situ Stress & UIC Class VI
        self.item_uic = QTreeWidgetItem(["In-Situ Stress & EPA Class VI Ceiling"])
        self.item_uic.setIcon(0, create_subsurface_icon("uic"))
        self.item_uic.setData(0, Qt.ItemDataRole.UserRole, ("geomechanics", "uic"))

        item_uic_tbl = QTreeWidgetItem(["EPA Class VI Safe Injection Limits (Table)"])
        item_uic_tbl.setIcon(0, create_subsurface_icon("table"))
        item_uic_tbl.setData(0, Qt.ItemDataRole.UserRole, ("geomechanics", "stress_table"))
        self.item_uic.addChild(item_uic_tbl)

        item_epa_comp = QTreeWidgetItem(["EPA Class VI Compliance Master Audit (Table)"])
        item_epa_comp.setIcon(0, create_subsurface_icon("table"))
        item_epa_comp.setData(0, Qt.ItemDataRole.UserRole, ("geomechanics", "epa_compliance_table"))
        self.item_uic.addChild(item_epa_comp)

        item_uic_grp = QTreeWidgetItem(["In-Situ Stress & UIC Pressure Gradients (Graph)"])
        item_uic_grp.setIcon(0, create_subsurface_icon("graph"))
        item_uic_grp.setData(0, Qt.ItemDataRole.UserRole, ("geomechanics", "stress_graph"))
        self.item_uic.addChild(item_uic_grp)

        item_aor = QTreeWidgetItem(["Area of Review (AoR) Plume & USDW Margin (Graph)"])
        item_aor.setIcon(0, create_subsurface_icon("graph"))
        item_aor.setData(0, Qt.ItemDataRole.UserRole, ("geomechanics", "aor_plume_graph"))
        self.item_uic.addChild(item_aor)

        item_spath = QTreeWidgetItem(["Dynamic Stress Path & Mohr Envelope (Graph)"])
        item_spath.setIcon(0, create_subsurface_icon("graph"))
        item_spath.setData(0, Qt.ItemDataRole.UserRole, ("geomechanics", "stress_path_graph"))
        self.item_uic.addChild(item_spath)
        self.geo_root.addChild(self.item_uic)

        self.tree.addTopLevelItem(self.geo_root)

        # 6. Storage, Utilisation & Geothermal Domain
        self.storage_root = QTreeWidgetItem(["Storage, Utilisation & Geothermal"])
        self.storage_root.setIcon(0, create_subsurface_icon("storage"))
        self.storage_root.setData(0, Qt.ItemDataRole.UserRole, ("storage", "root"))

        # Parent item: CCUS Storage Trapping Mechanisms
        self.item_trap = QTreeWidgetItem(["CCUS Storage Trapping Mechanisms"])
        self.item_trap.setIcon(0, create_subsurface_icon("storage"))
        self.item_trap.setData(0, Qt.ItemDataRole.UserRole, ("storage", "trapping"))

        item_trap_tbl = QTreeWidgetItem(["CCUS Trapping Mechanisms Breakdown (Table)"])
        item_trap_tbl.setIcon(0, create_subsurface_icon("table"))
        item_trap_tbl.setData(0, Qt.ItemDataRole.UserRole, ("storage", "trapping_table"))
        self.item_trap.addChild(item_trap_tbl)

        item_trap_grp = QTreeWidgetItem(["Dynamic CO2 Trapping Evolution (Graph)"])
        item_trap_grp.setIcon(0, create_subsurface_icon("graph"))
        item_trap_grp.setData(0, Qt.ItemDataRole.UserRole, ("storage", "trapping_graph"))
        self.item_trap.addChild(item_trap_grp)
        self.storage_root.addChild(self.item_trap)

        # Parent item: Section 45Q & Carbon Utilization
        self.item_econ = QTreeWidgetItem(["Section 45Q & Carbon Utilization"])
        self.item_econ.setIcon(0, create_subsurface_icon("volumetrics"))
        self.item_econ.setData(0, Qt.ItemDataRole.UserRole, ("storage", "economics"))

        item_econ_tbl = QTreeWidgetItem(["Section 45Q Tax Credit & Utilization Accounting (Table)"])
        item_econ_tbl.setIcon(0, create_subsurface_icon("table"))
        item_econ_tbl.setData(0, Qt.ItemDataRole.UserRole, ("storage", "economics_table"))
        self.item_econ.addChild(item_econ_tbl)
        self.storage_root.addChild(self.item_econ)

        # Parent item: CO2 Plume Geothermal (CPG) Energy Recovery
        self.item_geo = QTreeWidgetItem(["CO2 Plume Geothermal (CPG) Energy"])
        self.item_geo.setIcon(0, create_subsurface_icon("geothermal"))
        self.item_geo.setData(0, Qt.ItemDataRole.UserRole, ("storage", "geothermal"))

        item_geo_tbl = QTreeWidgetItem(["CPG Thermal Energy & Baseload Power Balance (Table)"])
        item_geo_tbl.setIcon(0, create_subsurface_icon("table"))
        item_geo_tbl.setData(0, Qt.ItemDataRole.UserRole, ("storage", "geothermal_table"))
        self.item_geo.addChild(item_geo_tbl)

        item_geo_grp = QTreeWidgetItem(["CPG Power Output & Heat Extraction (Graph)"])
        item_geo_grp.setIcon(0, create_subsurface_icon("graph"))
        item_geo_grp.setData(0, Qt.ItemDataRole.UserRole, ("storage", "geothermal_graph"))
        self.item_geo.addChild(item_geo_grp)
        self.storage_root.addChild(self.item_geo)

        self.tree.addTopLevelItem(self.storage_root)

        # 7. Surveillance & Quality Audit
        self.surv_root = QTreeWidgetItem(["Surveillance & Quality Audit"])
        self.surv_root.setIcon(0, create_subsurface_icon("audit"))
        self.surv_root.setData(0, Qt.ItemDataRole.UserRole, ("surveillance", "root"))

        item_audit_gate = QTreeWidgetItem(["Pre-Flight Model Audit Checklist (Table)"])
        item_audit_gate.setIcon(0, create_subsurface_icon("table"))
        item_audit_gate.setData(0, Qt.ItemDataRole.UserRole, ("surveillance", "audit_gate"))
        self.surv_root.addChild(item_audit_gate)

        item_multi_domain = QTreeWidgetItem(["Multi-Domain Surveillance Workstation (Dashboard)"])
        item_multi_domain.setIcon(0, create_subsurface_icon("workstation"))
        item_multi_domain.setData(0, Qt.ItemDataRole.UserRole, ("surveillance", "multi_domain"))
        self.surv_root.addChild(item_multi_domain)

        self.tree.addTopLevelItem(self.surv_root)

        # Expand all top level
        for i in range(self.tree.topLevelItemCount()):
            self.tree.topLevelItem(i).setExpanded(True)

    def update_wells_list(self, well_data_list):
        """Updates the Well Network tree branch with loaded wells and attached graphs/sheets."""
        self.wells_root.takeChildren()

        # Overview items
        item_winv = QTreeWidgetItem(["Master Well Inventory & Overview (Table)"])
        item_winv.setIcon(0, create_subsurface_icon("table"))
        item_winv.setData(0, Qt.ItemDataRole.UserRole, ("wells", "inventory_table"))
        self.wells_root.addChild(item_winv)

        item_wsched = QTreeWidgetItem(["Schedule & Operating Controls (Table)"])
        item_wsched.setIcon(0, create_subsurface_icon("schedule"))
        item_wsched.setData(0, Qt.ItemDataRole.UserRole, ("wells", "schedule_table"))
        self.wells_root.addChild(item_wsched)

        item_wgantt = QTreeWidgetItem(["Well Lifecycle & Development Timeline (Graph)"])
        item_wgantt.setIcon(0, create_subsurface_icon("gantt"))
        item_wgantt.setData(0, Qt.ItemDataRole.UserRole, ("wells", "schedule_gantt"))
        self.wells_root.addChild(item_wgantt)

        item_wwag = QTreeWidgetItem(["WAG Alternating Injection Schedule (Graph)"])
        item_wwag.setIcon(0, create_subsurface_icon("graph"))
        item_wwag.setData(0, Qt.ItemDataRole.UserRole, ("wells", "wag_schedule_graph"))
        self.wells_root.addChild(item_wwag)

        item_wsweep = QTreeWidgetItem(["Inter-Well Sweep & Connectivity (Graph)"])
        item_wsweep.setIcon(0, create_subsurface_icon("graph"))
        item_wsweep.setData(0, Qt.ItemDataRole.UserRole, ("wells", "sweep_matrix"))
        self.wells_root.addChild(item_wsweep)

        if not well_data_list:
            empty_item = QTreeWidgetItem(["No wells (Click Add Well)"])
            empty_item.setIcon(0, create_subsurface_icon("wells"))
            empty_item.setData(0, Qt.ItemDataRole.UserRole, ("wells", "empty"))
            self.wells_root.addChild(empty_item)
            return

        for well in well_data_list:
            w_name = getattr(well, "name", "Well")
            w_type = str(getattr(well, "metadata", {}).get("type", "")).lower()
            if not w_type:
                w_type = "injector" if "inj" in w_name.lower() else "producer"
            is_inj = "inj" in w_type
            tag = "INJ" if is_inj else "PROD"
            perfs_count = len(getattr(well, "perforations", []) or getattr(well, "perforation_properties", []))
            
            w_item = QTreeWidgetItem([f"[{tag}] {w_name} ({perfs_count} perfs)"])
            w_item.setIcon(0, create_subsurface_icon("well_inj" if is_inj else "well_prod"))
            w_item.setData(0, Qt.ItemDataRole.UserRole, ("well_item", w_name))

            # Attached children inside well item:
            item_w_traj = QTreeWidgetItem(["Trajectory & Deviation Survey (Table)"])
            item_w_traj.setIcon(0, create_subsurface_icon("table"))
            item_w_traj.setData(0, Qt.ItemDataRole.UserRole, ("well_item", f"{w_name}:trajectory_table"))
            w_item.addChild(item_w_traj)

            item_w_ipr = QTreeWidgetItem(["Inflow Performance IPR Curve (Graph)"])
            item_w_ipr.setIcon(0, create_subsurface_icon("graph"))
            item_w_ipr.setData(0, Qt.ItemDataRole.UserRole, ("well_item", f"{w_name}:ipr_graph"))
            w_item.addChild(item_w_ipr)

            item_w_prof = QTreeWidgetItem(["Wellbore Profile & Formation Section (Graph)"])
            item_w_prof.setIcon(0, create_subsurface_icon("graph"))
            item_w_prof.setData(0, Qt.ItemDataRole.UserRole, ("well_item", f"{w_name}:profile_graph"))
            w_item.addChild(item_w_prof)

            item_w_sens = QTreeWidgetItem(["Peaceman WI & Skin Sensitivity (Graph)"])
            item_w_sens.setIcon(0, create_subsurface_icon("graph"))
            item_w_sens.setData(0, Qt.ItemDataRole.UserRole, ("well_item", f"{w_name}:sensitivity_graph"))
            w_item.addChild(item_w_sens)

            item_w_draw = QTreeWidgetItem(["Radial Pressure Drawdown Cone (Graph)"])
            item_w_draw.setIcon(0, create_subsurface_icon("graph"))
            item_w_draw.setData(0, Qt.ItemDataRole.UserRole, ("well_item", f"{w_name}:drawdown_graph"))
            w_item.addChild(item_w_draw)

            self.wells_root.addChild(w_item)
            
        self.wells_root.setExpanded(True)

    def update_faults_list(self, fault_data_list):
        """Updates the Faults tree branch with dynamic fault items, geometry, isolated 3D view, and slip graphs."""
        self.item_faults.takeChildren()

        item_flt_ov = QTreeWidgetItem(["Master Fault System Overview (Table)"])
        item_flt_ov.setIcon(0, create_subsurface_icon("table"))
        item_flt_ov.setData(0, Qt.ItemDataRole.UserRole, ("geomechanics", "fault_table"))
        self.item_faults.addChild(item_flt_ov)

        item_cfs = QTreeWidgetItem(["Inter-Fault Stress Transfer Matrix (Graph)"])
        item_cfs.setIcon(0, create_subsurface_icon("graph"))
        item_cfs.setData(0, Qt.ItemDataRole.UserRole, ("geomechanics", "fault_inter_stress"))
        self.item_faults.addChild(item_cfs)

        if not fault_data_list:
            empty_item = QTreeWidgetItem(["No faults defined (Click Add Fault)"])
            empty_item.setIcon(0, create_subsurface_icon("faults"))
            empty_item.setData(0, Qt.ItemDataRole.UserRole, ("geomechanics", "empty"))
            self.item_faults.addChild(empty_item)
            return

        for f in fault_data_list:
            f_id = getattr(f, "id", "F-1")
            f_name = getattr(f, "name", "Fault")
            f_throw = float(getattr(f, "throw", 0.0))
            f_strike = float(getattr(f, "strike", 45.0))
            f_ts = float(getattr(f, "slip_tendency", 0.42))

            flt_item = QTreeWidgetItem([f"[{f_id}] {f_name} (Throw: {f_throw:+.0f} ft, Ts: {f_ts:.2f})"])
            flt_item.setIcon(0, create_subsurface_icon("faults"))
            flt_item.setData(0, Qt.ItemDataRole.UserRole, ("fault_item", f_name))

            item_f_geo = QTreeWidgetItem(["Geometry & Kinematics (Table)"])
            item_f_geo.setIcon(0, create_subsurface_icon("table"))
            item_f_geo.setData(0, Qt.ItemDataRole.UserRole, ("fault_item", f"{f_name}:geometry_table"))
            flt_item.addChild(item_f_geo)

            item_f_3d = QTreeWidgetItem(["Isolated 3D Fault Surface (3D View)"])
            item_f_3d.setIcon(0, create_subsurface_icon("faults"))
            item_f_3d.setData(0, Qt.ItemDataRole.UserRole, ("fault_item", f"{f_name}:3d_isolated"))
            flt_item.addChild(item_f_3d)

            item_f_slip = QTreeWidgetItem(["Slip Tendency & Reactivation (Graph)"])
            item_f_slip.setIcon(0, create_subsurface_icon("graph"))
            item_f_slip.setData(0, Qt.ItemDataRole.UserRole, ("fault_item", f"{f_name}:slip_graph"))
            flt_item.addChild(item_f_slip)

            item_f_juxt = QTreeWidgetItem(["Juxtaposition & SGR Profile (Graph)"])
            item_f_juxt.setIcon(0, create_subsurface_icon("graph"))
            item_f_juxt.setData(0, Qt.ItemDataRole.UserRole, ("fault_item", f"{f_name}:sgr_graph"))
            flt_item.addChild(item_f_juxt)

            self.item_faults.addChild(flt_item)

        self.item_faults.setExpanded(True)

    def select_well_node(self, well_name: str):
        """Selects the well node in the tree corresponding to a clicked 3D well without re-emitting signals."""
        for i in range(self.wells_root.childCount()):
            child = self.wells_root.child(i)
            data = child.data(0, Qt.ItemDataRole.UserRole)
            if data and data[0] == "well_item" and (data[1] == well_name or str(data[1]).startswith(f"{well_name}:")):
                self.tree.blockSignals(True)
                self.tree.setCurrentItem(child)
                self.tree.blockSignals(False)
                break

    def select_domain_item(self, domain: str, item_key: str, emit_signal: bool = False):
        """Highlights the tree item corresponding to a domain and item key without recursive signal loops."""
        def search_node(item: QTreeWidgetItem) -> bool:
            data = item.data(0, Qt.ItemDataRole.UserRole)
            if data and data[0] == domain and data[1] == item_key:
                self.tree.blockSignals(True)
                self.tree.setCurrentItem(item)
                self.tree.blockSignals(False)
                if emit_signal:
                    self.node_selected.emit(domain, item_key)
                return True
            for i in range(item.childCount()):
                if search_node(item.child(i)):
                    return True
            return False

        for idx in range(self.tree.topLevelItemCount()):
            if search_node(self.tree.topLevelItem(idx)):
                break

    def _on_item_clicked(self, item: QTreeWidgetItem, column: int):
        data = item.data(0, Qt.ItemDataRole.UserRole)
        if data:
            domain, item_key = data
            logger.debug(f"ModelTreeWidget item clicked: domain='{domain}', key='{item_key}'")
            self.node_selected.emit(domain, item_key)

    def _on_current_item_changed(self, current: Optional[QTreeWidgetItem], previous: Optional[QTreeWidgetItem]):
        if current:
            data = current.data(0, Qt.ItemDataRole.UserRole)
            if data:
                domain, item_key = data
                self.node_selected.emit(domain, item_key)

    def _show_tree_context_menu(self, pos):
        """Displays rich right-click context menu for selected tree node."""
        item = self.tree.itemAt(pos)
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

        data = item.data(0, Qt.ItemDataRole.UserRole) if item else None
        if data:
            domain, item_key = data
            if domain == "well_item":
                act_inspect = menu.addAction(f"Inspect Properties ({item_key})")
                act_inspect.triggered.connect(lambda: self.node_selected.emit(domain, item_key))
                act_add_perf = menu.addAction("Add Perforation Interval...")
                act_add_perf.triggered.connect(lambda: self.node_selected.emit(domain, item_key))
                menu.addSeparator()
            elif domain == "wells":
                act_add = menu.addAction("Add New Well...")
                act_add.triggered.connect(self.add_well_requested.emit)
                act_5spot = menu.addAction("Generate Standard 5-Spot Pattern")
                act_5spot.triggered.connect(lambda: self.generate_pattern_requested.emit("5spot"))
                act_9spot = menu.addAction("Generate Inverted 9-Spot Pattern")
                act_9spot.triggered.connect(lambda: self.generate_pattern_requested.emit("9spot"))
                menu.addSeparator()
            elif domain == "reservoir":
                act_select = menu.addAction(f"Configure {item.text(0)}")
                act_select.triggered.connect(lambda: self.node_selected.emit(domain, item_key))
                act_strat = menu.addAction("Open Stratigraphy Cross-Section")
                act_strat.triggered.connect(lambda: self.node_selected.emit("reservoir", "stratigraphy"))
                menu.addSeparator()
            elif domain == "geomechanics":
                act_cap = menu.addAction("Inspect Caprock Seal & Containment")
                act_cap.triggered.connect(lambda: self.node_selected.emit("geomechanics", "caprock"))
                act_fault = menu.addAction("Inspect Fault Slip Tendency")
                act_fault.triggered.connect(lambda: self.node_selected.emit("geomechanics", "faults"))
                menu.addSeparator()
            elif domain == "surveillance":
                act_audit = menu.addAction("Run Pre-Flight Visual Audit Gate")
                act_audit.triggered.connect(lambda: self.node_selected.emit("surveillance", "audit"))
                act_multi = menu.addAction("Open Multi-Domain Surveillance")
                act_multi.triggered.connect(lambda: self.node_selected.emit("surveillance", "multi_domain"))
                menu.addSeparator()

        act_expand_all = menu.addAction("Expand All Branches")
        act_expand_all.triggered.connect(self.tree.expandAll)

        act_collapse_all = menu.addAction("Collapse All Branches")
        act_collapse_all.triggered.connect(self.tree.collapseAll)

        menu.exec(self.tree.mapToGlobal(pos))
