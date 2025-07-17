from PyQt6.QtWidgets import QPushButton, QWidget
from PyQt6.QtGui import QIcon, QCursor
from PyQt6.QtCore import Qt, pyqtSignal

class HelpIcon(QPushButton):
    """
    A standardized '?' icon that, when clicked, finds the main window
    and asks it to display help for a specific topic key.
    """
    # Define a signal that will be emitted with the help key
    help_requested = pyqtSignal(str)

    def __init__(self, help_key: str, parent: QWidget = None):
        super().__init__(QIcon.fromTheme("help-contextual"), "", parent)
        if not self.icon():
             self.setText("?") # Fallback if theme icon is not found
        
        self.help_key = help_key
        self.setToolTip("Click for detailed help")
        self.setCursor(QCursor(Qt.CursorShape.WhatsThisCursor))
        self.setFixedSize(22, 22)
        self.setStyleSheet("QPushButton { border: none; border-radius: 11px; }")
        
        # Connect the button's click to our handler
        self.clicked.connect(self._request_help)

    def _request_help(self):
        """Find the main window and emit its help_requested signal."""
        main_win = self.window()
        if hasattr(main_win, 'request_help'):
            # This assumes the main window has a slot/method named 'request_help'
            main_win.request_help(self.help_key)
        else:
            print(f"Error: Could not find 'request_help' method on main window to ask for key: {self.help_key}")