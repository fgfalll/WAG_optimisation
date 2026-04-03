from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QMessageBox
import json

class ContextBubble(QWidget):
    def __init__(self, context_type, data):
        super().__init__()
        self.data = data
        
        layout = QVBoxLayout(self)
        
        # Token count (simple estimation)
        try:
            token_count = len(json.dumps(data).split())
        except TypeError:
            token_count = len(str(data).split())

        self.info_label = QLabel(f"Context received for '{context_type}'. (~{token_count} tokens)")
        layout.addWidget(self.info_label)
        
        self.actions_layout = QHBoxLayout()
        self.show_context_btn = QPushButton("Show Context")
        self.show_context_btn.clicked.connect(self._show_context)
        self.actions_layout.addWidget(self.show_context_btn)
        layout.addLayout(self.actions_layout)
        
        self.setStyleSheet("""
            background-color: #FFF9C4; /* Light yellow */
            border-radius: 10px;
            padding: 10px;
            border: 1px solid #FBC02D;
        """)

    def _show_context(self):
        dialog = QMessageBox(self)
        dialog.setWindowTitle("Context Data")
        try:
            dialog.setText(json.dumps(self.data, indent=2))
        except TypeError:
            dialog.setText(str(self.data))
        dialog.exec()
