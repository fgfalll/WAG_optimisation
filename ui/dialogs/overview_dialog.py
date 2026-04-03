from PyQt6.QtWidgets import QWidget, QVBoxLayout, QTextBrowser

class OverviewWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        
        text_browser = QTextBrowser()
        text_browser.setOpenExternalLinks(True)
        layout.addWidget(text_browser)

        # Read the content from the Markdown file
        try:
            with open('doc/COMPREHENSIVE_SYSTEM_OVERVIEW.md', 'r', encoding='utf-8') as f:
                markdown_content = f.read()
            text_browser.setMarkdown(markdown_content)
        except FileNotFoundError:
            text_browser.setText("Could not find the overview documentation file.")
