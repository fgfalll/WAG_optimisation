import logging
from PyQt6.QtWidgets import QWidget, QHBoxLayout, QPushButton, QLabel, QFrame
from PyQt6.QtGui import QIcon
from PyQt6.QtCore import pyqtSignal

logger = logging.getLogger(__name__)


class QuickPresetBar(QFrame):
    """
    Top ribbon bar providing 1-click field benchmark loaders and quick project actions.
    Ensures zero-data accessibility by allowing users to instantly load realistic models.
    """
    preset_selected = pyqtSignal(str)     # "spe5", "permian", "weyburn", "gulf_coast"
    sync_requested = pyqtSignal()
    audit_requested = pyqtSignal()
    visual_audit_requested = pyqtSignal()
    workstation_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setStyleSheet("""
            QFrame {
                background: #1e222b;
                border-bottom: 1px solid #2d3340;
                padding: 3px 6px;
            }
            QLabel {
                color: #8c9ba5;
                font-size: 11px;
                font-weight: bold;
            }
            QPushButton {
                background: #2a303c;
                color: #e1e7ec;
                border: 1px solid #3c4454;
                border-radius: 4px;
                padding: 4px 10px;
                font-size: 11px;
                font-weight: 500;
            }
            QPushButton:hover {
                background: #363d4d;
                border-color: #4e596f;
            }
            QPushButton#btnSync {
                background: #0d6efd;
                color: white;
                border: 1px solid #0b5ed7;
                font-weight: bold;
            }
            QPushButton#btnSync:hover {
                background: #0b5ed7;
            }
        """)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 4, 6, 4)
        layout.setSpacing(6)

        # Benchmark templates
        lbl_templates = QLabel("Field Presets:")
        layout.addWidget(lbl_templates)

        self.btn_spe5 = QPushButton("SPE 5 Benchmark")
        self.btn_spe5.setToolTip("Load SPE 5 3-layer cross-flow CO2 miscible flood benchmark")
        self.btn_spe5.clicked.connect(lambda: self.preset_selected.emit("spe5"))
        layout.addWidget(self.btn_spe5)

        self.btn_permian = QPushButton("Permian Basin")
        self.btn_permian.setToolTip("Load Permian Basin San Andres carbonate reservoir parameters")
        self.btn_permian.clicked.connect(lambda: self.preset_selected.emit("permian"))
        layout.addWidget(self.btn_permian)

        self.btn_weyburn = QPushButton("Weyburn CO2 EOR")
        self.btn_weyburn.setToolTip("Load Midale carbonate Weyburn field EOR benchmark model")
        self.btn_weyburn.clicked.connect(lambda: self.preset_selected.emit("weyburn"))
        layout.addWidget(self.btn_weyburn)

        layout.addStretch()

        # Action shortcuts
        self.btn_audit = QPushButton("Pre-Flight Physical Audit")
        self.btn_audit.setToolTip("Run physics, material balance, and geomechanical Class VI verification")
        self.btn_audit.clicked.connect(self.audit_requested.emit)
        layout.addWidget(self.btn_audit)

        self.btn_visual_audit = QPushButton("Visual Audit Gate")
        self.btn_visual_audit.setToolTip("Open Shared Earth visual inspection and confirmation gate")
        self.btn_visual_audit.clicked.connect(self.visual_audit_requested.emit)
        layout.addWidget(self.btn_visual_audit)

        self.btn_workstation = QPushButton("Model Workstation")
        self.btn_workstation.setToolTip("Open full-scale multi-domain Shared Earth Model evaluation dashboard")
        self.btn_workstation.clicked.connect(self.workstation_requested.emit)
        layout.addWidget(self.btn_workstation)

        self.btn_sync = QPushButton("Sync & Generate Data")
        self.btn_sync.setObjectName("btnSync")
        self.btn_sync.setToolTip("Compile input parameters, calculate grid properties, and synchronize across optimizer")
        self.btn_sync.clicked.connect(self.sync_requested.emit)
        layout.addWidget(self.btn_sync)
