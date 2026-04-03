import logging
from typing import Optional

from PyQt6.QtWidgets import (
    QFrame, QVBoxLayout, QTextBrowser, QLabel, QPushButton, QWidget, QHBoxLayout
)
from PyQt6.QtGui import QIcon
from PyQt6.QtCore import pyqtSignal, pyqtSlot, QEvent
from help_manager import HelpManager

logger = logging.getLogger(__name__)

class HelpPanel(QFrame):
    closed = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None):
        """Initializes the help panel, initially hidden."""
        super().__init__(parent)
        self.help_manager = HelpManager()
        self._current_error_key: Optional[str] = None
        self._is_showing_error: bool = False

        self.setObjectName("HelpPanel")
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setMinimumWidth(350)
        self.setMaximumWidth(600)

        # --- UI Setup ---
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 5, 10, 10)
        main_layout.setSpacing(5)
        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(0, 0, 0, 0)
        
        self.title_label = QLabel()
        self.title_label.setStyleSheet("font-size: 14pt; font-weight: bold; color: #333;")
        
        self.close_btn = QPushButton(QIcon.fromTheme("window-close"), "")
        self.close_btn.setFlat(True)
        self.close_btn.clicked.connect(self.hide)
        
        header_layout.addWidget(self.title_label)
        header_layout.addStretch()
        header_layout.addWidget(self.close_btn)

        # Main content area for displaying formatted HTML
        self.help_browser = QTextBrowser()
        self.help_browser.setOpenExternalLinks(True)
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
        self.retranslateUi()
        self.hide()
    
    def retranslateUi(self):
        self.close_btn.setToolTip(self.tr("Close Help Panel"))
        
        if self._is_showing_error:
            self._display_error_message()
        else:
            if not self.help_browser.documentTitle():
                 self.title_label.setText(self.tr("Help"))

    def changeEvent(self, event: QEvent):
        """Handles events, specifically for language changes."""
        if event.type() == QEvent.Type.LanguageChange:
            self.retranslateUi()
        super().changeEvent(event)

    def _display_error_message(self):
        """Displays a formatted, translatable error message."""
        self.title_label.setText(self.tr("Help Not Found"))
        self.help_browser.setHtml(
            f"<h1>{self.tr('Topic Not Found')}</h1>"
            f"<p>{self.tr("Detailed help for the key '{key}' is not available.").format(key=f'<code>{self._current_error_key}</code>')}</p>"
            f"<p>{self.tr('Please check that this key is correctly defined in the <b>config/help_content.yaml</b> file and that its associated Markdown file exists.')}</p>"
        )

    @pyqtSlot(str)
    def show_help_for(self, parameter_key: str):
        if not parameter_key:
            return

        content = self.help_manager.get_page_content_for_key(parameter_key)
        
        if content is None:
            logger.warning(f"No help content could be retrieved for key: {parameter_key}")
            self._is_showing_error = True
            self._current_error_key = parameter_key
            self._display_error_message()
            self.show()
            self.raise_()
            return
            
        self._is_showing_error = False
        self._current_error_key = None
        page_title, html_content, anchor_id = content
        is_already_loaded = self.help_browser.documentTitle() == page_title
        self.title_label.setText(page_title)

        if not is_already_loaded:
            self.help_browser.setHtml(html_content)
            self.help_browser.setDocumentTitle(page_title)
        
        if anchor_id:
            self.help_browser.scrollToAnchor(anchor_id)
        else:
            self.help_browser.verticalScrollBar().setValue(0)
        
        self.show()
        self.raise_()

    def hideEvent(self, event):
        self.closed.emit()
        super().hideEvent(event)