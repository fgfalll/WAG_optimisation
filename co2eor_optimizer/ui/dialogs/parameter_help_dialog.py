import logging
from typing import Optional

from PyQt6.QtWidgets import (
    QFrame, QVBoxLayout, QTextBrowser, QLabel, QPushButton, QWidget, QHBoxLayout
)
from PyQt6.QtGui import QIcon
from PyQt6.QtCore import pyqtSignal, Qt, pyqtSlot

# This import works because help_manager.py is in the co2eor_optimizer package root,
# making it accessible via an absolute import path from anywhere inside the package.
from co2eor_optimizer.help_manager import HelpManager

logger = logging.getLogger(__name__)

class HelpPanel(QFrame):
    """
    A non-modal side panel to display detailed, scrollable help information.
    It fetches rich text content from the HelpManager based on a parameter key
    and can scroll to specific anchors within a help page.
    """
    # Signal emitted when the panel is hidden by the user's actions.
    closed = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None):
        """Initializes the help panel, initially hidden."""
        super().__init__(parent)
        self.help_manager = HelpManager()
        self.setObjectName("HelpPanel")
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setMinimumWidth(350)
        self.setMaximumWidth(600)  # A sensible max width to prevent it from taking over.

        # --- UI Setup ---
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 5, 10, 10)
        main_layout.setSpacing(5)

        # Header section with title and close button
        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(0, 0, 0, 0)
        
        self.title_label = QLabel("Help")
        self.title_label.setStyleSheet("font-size: 14pt; font-weight: bold; color: #333;")
        
        close_btn = QPushButton(QIcon.fromTheme("window-close"), "")
        close_btn.setToolTip("Close Help Panel")
        close_btn.setFlat(True)
        close_btn.clicked.connect(self.hide) # Connect to the widget's hide method
        
        header_layout.addWidget(self.title_label)
        header_layout.addStretch()
        header_layout.addWidget(close_btn)

        # Main content area for displaying formatted HTML
        self.help_browser = QTextBrowser()
        self.help_browser.setOpenExternalLinks(True)  # Allows opening http:// links
        
        # Apply styling for better readability of the help content.
        self.help_browser.setStyleSheet("""
            QTextBrowser {
                background-color: #fcfcfc;
                border: 1px solid #ccc;
                border-radius: 4px;
                font-size: 10pt;
            }
            h1, h2, h3 { 
                margin-top: 10px; 
                margin-bottom: 5px; 
                border-bottom: 1px solid #eee; 
                padding-bottom: 3px;
            }
            p { 
                margin-top: 0; 
                line-height: 150%;
            }
            code { 
                background-color: #e8e8e8; 
                padding: 2px 4px; 
                border-radius: 3px; 
                font-family: Consolas, monospace;
            }
            ul {
                margin-left: 20px;
            }
        """)

        main_layout.addLayout(header_layout)
        main_layout.addWidget(self.help_browser)

        # Start hidden until a help topic is requested
        self.hide()

    @pyqtSlot(str)
    def show_help_for(self, parameter_key: str):
        """
        Public slot to show and populate the panel for a given parameter key.
        This is the main entry point for displaying help content.
        
        Args:
            parameter_key: The unique key, e.g., 'ConfigWidget.EconomicParameters.interest_rate'.
        """
        if not parameter_key:
            return

        content = self.help_manager.get_page_content_for_key(parameter_key)
        
        # Handle cases where the help key is invalid or the file is missing
        if content is None:
            logger.warning(f"No help content could be retrieved for key: {parameter_key}")
            self.title_label.setText("Help Not Found")
            self.help_browser.setHtml(
                f"<h1>Topic Not Found</h1>"
                f"<p>Detailed help for the key '<code>{parameter_key}</code>' is not available.</p>"
                f"<p>Please check that this key is correctly defined in the "
                f"<b>config/help_content.yaml</b> file and that its associated "
                f"Markdown file exists.</p>"
            )
            self.show()
            self.raise_()
            return

        # On success, populate the panel
        page_title, html_content, anchor_id = content
        
        # Optimization: Only reload HTML if the page is different
        is_already_loaded = self.help_browser.documentTitle() == page_title
        
        self.title_label.setText(page_title)

        if not is_already_loaded:
            self.help_browser.setHtml(html_content)
            # Use document title to track the currently loaded page
            self.help_browser.setDocumentTitle(page_title)
        
        # Scroll to the specific topic within the page
        if anchor_id:
            self.help_browser.scrollToAnchor(anchor_id)
        else:
            # If no anchor, just scroll to the top
            self.help_browser.verticalScrollBar().setValue(0)
        
        self.show()
        self.raise_() # Bring to the front

    def hideEvent(self, event):
        """
        Overrides the default hide event to emit a signal.
        This allows the MainWindow to react when the panel is closed.
        """
        self.closed.emit()
        super().hideEvent(event)