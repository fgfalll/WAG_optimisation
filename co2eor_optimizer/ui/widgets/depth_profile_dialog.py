from typing import List, Optional

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QDialogButtonBox, QWidget
)
from PyQt6.QtCore import QPointF

try:
    from .depth_profile_editor import DepthProfileEditor
except ImportError:
    class DepthProfileEditor(QWidget): pass

class DepthProfileDialog(QDialog):
    """
    A dedicated dialog window for interactively editing the well's depth profile.
    """
    def __init__(self, initial_path: List[QPointF], parent: Optional[QWidget] = None):
        """
        Initializes the dialog.
        
        Args:
            initial_path: The list of QPointF objects representing the current well path.
            parent: The parent widget.
        """
        super().__init__(parent)
        self.setWindowTitle("Edit Well Depth Profile")
        self.setMinimumSize(800, 600)
        self.setModal(True)

        layout = QVBoxLayout(self)
        
        # Create and initialize the editor widget
        self.editor = DepthProfileEditor(self)
        self.editor.set_path(initial_path)
        
        # Standard OK/Cancel buttons
        self.button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        
        layout.addWidget(self.editor)
        layout.addWidget(self.button_box)

    def get_path(self) -> List[QPointF]:
        """
        Returns the final, edited path from the editor widget.
        
        Returns:
            A list of QPointF objects for the edited path.
        """
        return self.editor.get_path()