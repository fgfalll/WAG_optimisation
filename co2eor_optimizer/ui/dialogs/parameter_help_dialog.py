import logging
from typing import Optional
from html import escape

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QTextBrowser, QDialogButtonBox, QSizePolicy, QWidget
)
from PyQt6.QtCore import Qt

logger = logging.getLogger(__name__)

class ParameterHelpDialog(QDialog):
    """
    A dialog to display detailed help information for a specific parameter,
    with support for basic HTML formatting in the content.
    """
    def __init__(self, parameter_name: str, content_html: str, parent: Optional[QWidget] = None):
        """
        Args:
            parameter_name: The display name of the parameter for the dialog title.
            content_html: The help content, formatted as HTML.
            parent: The parent widget.
        """
        super().__init__(parent)
        self.setWindowTitle(f"Help: {parameter_name}")
        self.setMinimumSize(450, 300)
        self.setWindowFlags(self.windowFlags() | Qt.WindowType.WindowSystemMenuHint | Qt.WindowType.WindowCloseButtonHint)

        layout = QVBoxLayout(self)

        help_browser = QTextBrowser()
        help_browser.setOpenExternalLinks(True)
        help_browser.setHtml(content_html)
        layout.addWidget(help_browser)

        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok)
        button_box.accepted.connect(self.accept)
        layout.addWidget(button_box)

    @staticmethod
    def show_help(parameter_name: str, help_content: str, parent: Optional[QWidget] = None):
        """
        Static method to conveniently create and show the help dialog.
        Automatically wraps plain text in paragraph tags for better formatting.

        Args:
            parameter_name: Display name for the dialog title.
            help_content: The help text (can be plain or HTML).
            parent: The parent widget for the dialog.
        """
        content_is_html = "<p>" in help_content or "<br" in help_content or "<h1>" in help_content
        if not content_is_html:
            # Escape plain text to prevent misinterpretation as HTML,
            # and convert newlines to <br> tags for proper line breaks.
            processed_content = f"<p>{escape(help_content).replace(chr(10), '<br>')}</p>"
        else:
            processed_content = help_content
            
        dialog = ParameterHelpDialog(parameter_name, processed_content, parent)
        dialog.exec()