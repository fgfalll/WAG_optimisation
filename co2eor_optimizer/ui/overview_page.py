import logging

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton, QApplication
)
from typing import Optional
from PyQt6.QtGui import QIcon, QFont
from PyQt6.QtCore import Qt, pyqtSignal, QSize

logger = logging.getLogger(__name__)

class OverviewPageWidget(QWidget):
    """
    The initial welcome screen for the application. It provides top-level
    actions to start a new project, load an existing one, or use a quick start demo.
    """
    start_new_project_requested = pyqtSignal()
    load_existing_project_requested = pyqtSignal()
    quick_start_requested = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("OverviewPage")
        self._setup_ui()
        self._connect_signals()
        self._apply_styling()

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(50, 50, 50, 50)
        main_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.setSpacing(20)

        main_layout.addStretch(2)

        # --- Application Title ---
        app_name = QApplication.applicationName() or "CO₂ EOR Suite"
        app_version = QApplication.applicationVersion() or ""

        title_label = QLabel(app_name)
        title_font = QFont()
        title_font.setPointSize(28)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(title_label)

        if app_version:
            version_label = QLabel(f"Version {app_version}")
            version_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            version_label.setObjectName("VersionLabel") # For styling
            main_layout.addWidget(version_label)

        main_layout.addSpacing(20)

        # --- Application Description ---
        description = (
            "This application is designed to help evaluate, simulate, and "
            "optimize CO₂-based Enhanced Oil Recovery (EOR) strategies."
        )
        description_label = QLabel(description)
        description_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        description_label.setWordWrap(True)
        description_label.setObjectName("DescriptionLabel")
        main_layout.addWidget(description_label)

        main_layout.addStretch(1)

        # --- Action Buttons ---
        button_layout = QVBoxLayout()
        button_layout.setSpacing(15)
        button_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.start_new_button = self._create_action_button(
            text=" Start New / Load Data",
            icon=QIcon.fromTheme("document-new"),
            tooltip="Begin a new analysis by loading your data files."
        )
        self.load_existing_button = self._create_action_button(
            text=" Load Project (.tphd)",
            icon=QIcon.fromTheme("document-open"),
            tooltip="Open a previously saved project file."
        )
        self.quick_start_button = self._create_action_button(
            text=" Quick Start Demo",
            icon=QIcon.fromTheme("go-next"),
            tooltip="Launch the application with pre-configured sample data."
        )

        button_layout.addWidget(self.start_new_button)
        button_layout.addWidget(self.load_existing_button)
        button_layout.addWidget(self.quick_start_button)

        main_layout.addLayout(button_layout)
        main_layout.addStretch(3)

    def _create_action_button(self, text: str, icon: QIcon, tooltip: str) -> QPushButton:
        """Helper method to create and style the main action buttons."""
        button = QPushButton(icon, text)
        button.setToolTip(tooltip)
        button.setMinimumSize(280, 50)
        button.setIconSize(QSize(24, 24))
        font = QFont()
        font.setPointSize(11)
        font.setBold(True)
        button.setFont(font)
        return button

    def _connect_signals(self):
        self.start_new_button.clicked.connect(self.start_new_project_requested)
        self.load_existing_button.clicked.connect(self.load_existing_project_requested)
        self.quick_start_button.clicked.connect(self.quick_start_requested)

    def _apply_styling(self):
        """Applies basic QSS styling. Can be overridden by a global stylesheet."""
        self.setStyleSheet("""
            #OverviewPage {
                background-color: #f8f9fa;
            }
            #VersionLabel {
                color: #6c757d;
            }
            #DescriptionLabel {
                font-size: 11pt;
                color: #495057;
            }
            QPushButton {
                background-color: #007bff;
                color: white;
                border: none;
                border-radius: 5px;
                padding: 10px;
            }
            QPushButton:hover {
                background-color: #0069d9;
            }
            QPushButton:pressed {
                background-color: #0056b3;
            }
        """)