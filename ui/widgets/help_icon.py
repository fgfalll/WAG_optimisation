from PyQt6.QtWidgets import QPushButton, QWidget
from PyQt6.QtGui import QIcon, QCursor
from PyQt6.QtCore import Qt, pyqtSignal, QEvent

import logging

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
        self.setCursor(QCursor(Qt.CursorShape.WhatsThisCursor))
        self.setFixedSize(22, 22)
        self.setStyleSheet("QPushButton { border: none; border-radius: 11px; }")
        
        # Connect the button's click to our handler
        self.clicked.connect(self._request_help)

        # Set initial translatable text
        self.retranslateUi()

    def _request_help(self):
        """Find the main window and emit its help_requested signal."""
        main_win = self.window()
        if hasattr(main_win, 'request_help'):
            # This assumes the main window has a slot/method named 'request_help'
            main_win.request_help(self.help_key)
        else:
            logging.error(f"Error: Could not find 'request_help' method on main window to ask for key: {self.help_key}")

    def retranslateUi(self):
        """
        Updates the widget's text to the current application language.
        """
        self.setToolTip(self.tr("Click for detailed help"))

    def changeEvent(self, event: QEvent):
        """
        Handles events sent to the widget, specifically for language changes.
        """
        if event.type() == QEvent.Type.LanguageChange:
            self.retranslateUi()
        super().changeEvent(event)