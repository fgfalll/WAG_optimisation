"""
Calculation Progress Dialog.
Provides a modern, clear, non-intrusive progress modal during thermodynamic,
petrophysical, and reservoir calculation routines to keep the user informed.
"""

import time
import logging
from typing import Optional
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QProgressBar, QFrame, QApplication, QWidget
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont

logger = logging.getLogger(__name__)


class CalculationProgressDialog(QDialog):
    """
    Modern progress dialog displayed during subsurface model recalculations,
    thermodynamic EOS solutions, and project data synchronization.
    """

    def __init__(self, parent: Optional[QWidget] = None, title: str = "Recalculating Model...", task_name: str = "Thermodynamic & Subsurface Calculation"):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(True)
        self.setWindowModality(Qt.WindowModality.ApplicationModal)
        self.setWindowFlags(Qt.WindowType.Dialog | Qt.WindowType.CustomizeWindowHint | Qt.WindowType.WindowTitleHint)
        self.setFixedSize(450, 145)

        self._task_name = task_name
        self._setup_ui()
        self._center_on_parent()

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(14, 12, 14, 12)
        main_layout.setSpacing(8)

        self.setStyleSheet("""
            QDialog {
                background: #ffffff;
            }
        """)

        # 1. Header Row: Title & Subtitle
        header_layout = QVBoxLayout()
        header_layout.setSpacing(2)

        self.lbl_title = QLabel(f"⚡ {self._task_name}")
        self.lbl_title.setStyleSheet("font-size: 12.5px; font-weight: bold; color: #0f172a;")
        header_layout.addWidget(self.lbl_title)

        self.lbl_step = QLabel("Initializing calculation routine...")
        self.lbl_step.setStyleSheet("font-size: 11px; color: #475569;")
        header_layout.addWidget(self.lbl_step)

        main_layout.addLayout(header_layout)

        # 2. Modern Progress Bar
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.progress_bar.setFixedHeight(22)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #cbd5e1;
                border-radius: 4px;
                background-color: #f8fafc;
                text-align: center;
                font-size: 11px;
                font-weight: bold;
                color: #0f172a;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #0284c7, stop:1 #0ea5e9);
                border-radius: 3px;
            }
        """)
        main_layout.addWidget(self.progress_bar)

        # 3. Status Footer
        footer_layout = QHBoxLayout()
        footer_layout.setContentsMargins(0, 0, 0, 0)
        self.lbl_footer = QLabel("Please wait while the simulation model updates...")
        self.lbl_footer.setStyleSheet("font-size: 10px; color: #64748b; font-style: italic;")
        footer_layout.addWidget(self.lbl_footer)
        footer_layout.addStretch()

        self.lbl_pct = QLabel("0%")
        self.lbl_pct.setStyleSheet("font-size: 10.5px; font-weight: bold; color: #0284c7;")
        footer_layout.addWidget(self.lbl_pct)

        main_layout.addLayout(footer_layout)

    def _center_on_parent(self):
        parent = self.parentWidget()
        if parent:
            geo = parent.geometry()
            x = geo.x() + (geo.width() - self.width()) // 2
            y = geo.y() + (geo.height() - self.height()) // 2
            self.move(max(x, 50), max(y, 50))

    def set_step(self, progress_percent: int, message: str, delay_s: float = 0.0):
        """Updates progress bar value and status text with immediate UI refresh."""
        clamped_val = int(max(0, min(100, progress_percent)))
        self.progress_bar.setValue(clamped_val)
        self.lbl_pct.setText(f"{clamped_val}%")
        self.lbl_step.setText(message)
        QApplication.processEvents()
        if delay_s > 0:
            time.sleep(delay_s)
            QApplication.processEvents()

    def finish(self, completion_message: str = "✓ Calculation complete!", auto_close_ms: int = 120):
        """Marks calculation complete and closes dialog."""
        self.progress_bar.setValue(100)
        self.lbl_pct.setText("100%")
        self.lbl_step.setText(completion_message)
        QApplication.processEvents()
        if auto_close_ms > 0:
            QTimer.singleShot(auto_close_ms, self.accept)
        else:
            self.accept()

    def __enter__(self):
        self.show()
        QApplication.processEvents()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            self.finish()
        else:
            logger.error(f"Calculation failed inside progress context: {exc_val}")
            self.reject()
