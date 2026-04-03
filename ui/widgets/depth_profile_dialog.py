from typing import List, Optional

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QDialogButtonBox, QWidget
)
from PyQt6.QtCore import QPointF, QEvent

try:
    from .depth_profile_editor import DepthProfileEditor
except ImportError:
    class DepthProfileEditor(QWidget): pass

class DepthProfileDialog(QDialog):
    """A dialog for editing a well's depth profile via an interactive 2D editor."""
    def __init__(self, initial_path: List[QPointF], parent: Optional[QWidget] = None):

        super().__init__(parent)
        
        self.setMinimumSize(800, 600)
        self.setModal(True)

        layout = QVBoxLayout(self)

        self.editor = DepthProfileEditor(self)
        self.editor.set_path(initial_path)

        self.button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        
        layout.addWidget(self.editor)
        layout.addWidget(self.button_box)

        self.retranslateUi()

    def retranslateUi(self) -> None:
        """
        Updates the text of all UI elements to the current language.
        """
        self.setWindowTitle(self.tr("Edit Well Depth Profile"))

    def changeEvent(self, event: QEvent) -> None:
        """
        Handles events, specifically for dynamic language changes.
        """
        if event.type() == QEvent.Type.LanguageChange:
            self.retranslateUi()
        super().changeEvent(event)

    def get_path(self) -> List[QPointF]:
        return self.editor.get_path()