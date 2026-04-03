import markdown
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QTextBrowser, QPushButton, QSizePolicy
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QTextCursor

class ChatBubble(QWidget):
    request_copy = pyqtSignal()
    request_regenerate = pyqtSignal()
    request_edit = pyqtSignal()

    def __init__(self, text, is_user=False):
        super().__init__()
        self.is_user = is_user
        self.text = text
        
        self.layout = QVBoxLayout(self)
        self.text_browser = QTextBrowser()
        self.text_browser.setOpenExternalLinks(True)
        self.text_browser.setReadOnly(True)
        self.text_browser.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.text_browser.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.text_browser.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.text_browser.document().contentsChanged.connect(self._on_contents_changed)

        self.set_text(text)

        self.layout.addWidget(self.text_browser)
        
        # Actions
        self.actions_layout = QHBoxLayout()
        self.actions_layout.addStretch()
        self.copy_btn = QPushButton("Copy")
        self.copy_btn.clicked.connect(self.request_copy)
        self.actions_layout.addWidget(self.copy_btn)

        if not self.is_user:
            self.regenerate_btn = QPushButton("Regenerate")
            self.regenerate_btn.clicked.connect(self.request_regenerate)
            self.actions_layout.addWidget(self.regenerate_btn)
        else:
            self.edit_btn = QPushButton("Edit")
            self.edit_btn.clicked.connect(self.request_edit)
            self.actions_layout.addWidget(self.edit_btn)
            
        self.layout.addLayout(self.actions_layout)
        self.setLayout(self.layout)

        self._set_style()

    def _on_contents_changed(self):
        doc_height = self.text_browser.document().size().height()
        self.text_browser.setFixedHeight(int(doc_height) + 5) # Add a small margin

    def get_text(self) -> str:
        return self.text

    def set_text(self, text):
        self.text = text
        html = markdown.markdown(text, extensions=['fenced_code', 'tables'])
        self.text_browser.setHtml(html)

    def append_text(self, text):
        self.text += text
        html = markdown.markdown(self.text, extensions=['fenced_code', 'tables'])
        self.text_browser.setHtml(html)
        self.text_browser.moveCursor(QTextCursor.MoveOperation.End)

    def _set_style(self):
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        if self.is_user:
            self.setStyleSheet("""
                ChatBubble {
                    background-color: #E1F5FE;
                    border-radius: 15px;
                    padding: 10px;
                    border: 1px solid #B3E5FC;
                }
            """)
        else:
            self.setStyleSheet("""
                ChatBubble {
                    background-color: #FFFFFF;
                    border-radius: 15px;
                    padding: 10px;
                    border: 1px solid #E0E0E0;
                }
            """)
