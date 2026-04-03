"""
Engine Selection Widget
=======================

Simplified widget for selecting simulation engines for CO2-EOR optimization.
"""

import logging
from typing import Dict, Any, Optional
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QLabel,
    QComboBox,
    QPushButton,
    QGroupBox,
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont

logger = logging.getLogger(__name__)


class EngineSelectionWidget(QWidget):
    """Simplified widget for engine selection"""

    # Signals
    engine_changed = pyqtSignal(str)  # Emitted when engine selection changes
    engine_switch_requested = pyqtSignal(str)  # Emitted when user requests engine switch

    def __init__(self, parent=None):
        super().__init__(parent)
        self.available_engines = {}
        self.current_engine = None
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)

        # Engine selection group
        selection_group = QGroupBox("Select Engine")
        selection_layout = QGridLayout(selection_group)
        selection_layout.setColumnStretch(0, 0)
        selection_layout.setColumnStretch(1, 1)

        # Engine dropdown
        self.engine_label = QLabel("Engine:")
        self.engine_label.setMinimumWidth(100)
        self.engine_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.engine_combo = QComboBox()
        self.engine_combo.setMinimumWidth(150)
        self.engine_combo.currentTextChanged.connect(self.on_engine_selection_changed)
        selection_layout.addWidget(self.engine_label, 0, 0)
        selection_layout.addWidget(self.engine_combo, 0, 1)

        # Switch button
        self.switch_button = QPushButton("Switch Engine")
        self.switch_button.clicked.connect(self.switch_engine)
        self.switch_button.setEnabled(False)
        selection_layout.addWidget(self.switch_button, 1, 0, 1, 2)

        # Status
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("color: green; font-style: italic;")
        selection_layout.addWidget(self.status_label, 2, 0, 1, 2)

        layout.addWidget(selection_group)
        layout.addStretch()

        # Load available engines
        self.refresh_engines()

    def refresh_engines(self):
        """Refresh the list of available engines"""
        try:
            # Import here to avoid circular imports
            from core.engine_factory import EngineFactory

            self.available_engines = EngineFactory.get_available_engines()

            # Update combo box
            self.engine_combo.clear()

            if not self.available_engines:
                self.engine_combo.addItem("No engines available")
                self.switch_button.setEnabled(False)
                self.status_label.setText("No simulation engines available")
                self.status_label.setStyleSheet("color: red; font-style: italic;")
                return

            for engine_type, engine_info in self.available_engines.items():
                display_name = f"{engine_info.get('name', engine_type.title())} ({engine_type})"
                self.engine_combo.addItem(display_name, engine_type)

            self.switch_button.setEnabled(len(self.available_engines) > 1)
            self.status_label.setText(f"Found {len(self.available_engines)} engine(s)")
            self.status_label.setStyleSheet("color: green; font-style: italic;")

            # Select first engine by default
            if self.available_engines:
                self.engine_combo.setCurrentIndex(0)

        except Exception as e:
            logger.error(f"Failed to refresh engines: {e}")
            self.status_label.setText(f"Error loading engines: {e}")
            self.status_label.setStyleSheet("color: red; font-style: italic;")

    def on_engine_selection_changed(self, engine_type: str):
        """Handle engine selection change"""
        if not engine_type or engine_type == "No engines available":
            return

        # Get actual engine type from user data
        current_data = self.engine_combo.currentData()
        if current_data:
            self.engine_changed.emit(current_data)

    def switch_engine(self):
        """Switch to the selected engine"""
        current_data = self.engine_combo.currentData()
        if not current_data:
            return

        self.status_label.setText(f"Switching to {current_data} engine...")
        self.status_label.setStyleSheet("color: orange; font-style: italic;")
        self.engine_switch_requested.emit(current_data)

    def set_current_engine(self, engine_type: str):
        """Set the current engine type"""
        self.current_engine = engine_type

        # Update combo box selection
        for i in range(self.engine_combo.count()):
            if self.engine_combo.itemData(i) == engine_type:
                self.engine_combo.setCurrentIndex(i)
                break

    def get_selected_engine(self) -> Optional[str]:
        """Get the currently selected engine type"""
        return self.engine_combo.currentData()
