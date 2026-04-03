from PyQt6.QtWidgets import QWidget, QVBoxLayout, QPushButton, QSplitter
from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtCore import QUrl
import subprocess
import os

class BreakthroughPhysicsWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self._setup_ui()
        self.load_justification()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        splitter = QSplitter(self)
        layout.addWidget(splitter)

        self.justification_view = QWebEngineView()
        splitter.addWidget(self.justification_view)

        results_widget = QWidget()
        results_layout = QVBoxLayout(results_widget)
        self.run_tests_button = QPushButton(self.tr("Run Tests"))
        self.run_tests_button.clicked.connect(self.run_tests)
        results_layout.addWidget(self.run_tests_button)

        self.results_view = QWebEngineView()
        results_layout.addWidget(self.results_view)
        splitter.addWidget(results_widget)

    def load_justification(self):
        current_dir = os.path.dirname(os.path.realpath(__file__))
        html_file_path = os.path.join(current_dir, "..", "assets", "docs", "breakthrough_physics_justification.html")
        self.justification_view.setUrl(QUrl.fromLocalFile(html_file_path))

    def run_tests(self):
        self.results_view.setHtml("<h1>Running Breakthrough Physics Tests...</h1>")
        try:
            process = subprocess.run(["python", "tests/validate_breakthrough_physics.py"], capture_output=True, text=True, check=True, cwd="D:\\Стаття 2\\4.5\\co2eor_optimizer")
            output = process.stdout.replace("\n", "<br>")
            html = f"<h2>Test Results:</h2><pre>{output}</pre>"
            self.results_view.setHtml(html)
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            self.results_view.setHtml(f"<h1>Error running tests:</h1><pre>{e}</pre>")
