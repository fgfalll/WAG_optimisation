import logging
from typing import Optional, Any, Dict

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QGroupBox, QLabel, QLineEdit,
    QComboBox, QTextEdit, QTextBrowser, QPushButton, QMessageBox
)
from PyQt6.QtGui import QIcon
from PyQt6.QtCore import Qt, QSettings, pyqtSignal

from .workers.ai_query_worker import AIQueryWorker

logger = logging.getLogger(__name__)

AI_SERVICES_CONFIG = {
    "OpenAI": {"models": ["gpt-4o", "gpt-4-turbo"], "key_name": "openai_api_key", "url_name": "openai_base_url", "default_url": "https://api.openai.com/v1"},
    "Gemini": {"models": ["gemini-1.5-pro-latest", "gemini-1.5-flash-latest"], "key_name": "gemini_api_key"},
    "OpenRouter": {"models": ["google/gemini-flash-1.5", "openai/gpt-4o"], "key_name": "openrouter_api_key", "default_url": "https://openrouter.ai/api/v1"},
}

AI_CONTEXT_TASKS = [
    "General Query", "Explain Parameter", "Interpret Sensitivity Results", 
    "Interpret UQ Results", "Brainstorm EOR Strategy", "Draft Report Section"
]

class AIAssistantWidget(QWidget):
    """Widget for interacting with AI models for assistance and analysis."""
    request_context_data = pyqtSignal(str)

    def __init__(self, app_settings: QSettings, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.settings = app_settings
        self.worker: Optional[AIQueryWorker] = None
        self.context_data: Optional[Dict[str, Any]] = None
        self._setup_ui()
        self._load_service_config()

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        
        # --- API Config ---
        api_group = QGroupBox("AI Service Configuration")
        api_layout = QGridLayout(api_group)
        self.service_combo = QComboBox()
        self.service_combo.addItems(AI_SERVICES_CONFIG.keys())
        self.api_key_edit = QLineEdit()
        self.api_key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.show_key_btn = QPushButton(QIcon.fromTheme("visibility"), "")
        self.show_key_btn.setCheckable(True)
        self.base_url_edit = QLineEdit()
        save_api_btn = QPushButton(QIcon.fromTheme("document-save"), "Save API Configuration")
        
        api_layout.addWidget(QLabel("Service:"), 0, 0)
        api_layout.addWidget(self.service_combo, 0, 1, 1, 2)
        api_layout.addWidget(QLabel("API Key:"), 1, 0)
        api_layout.addWidget(self.api_key_edit, 1, 1)
        api_layout.addWidget(self.show_key_btn, 1, 2)
        self.base_url_label = QLabel("Base URL:")
        api_layout.addWidget(self.base_url_label, 2, 0)
        api_layout.addWidget(self.base_url_edit, 2, 1, 1, 2)
        api_layout.addWidget(save_api_btn, 3, 0, 1, 3)
        main_layout.addWidget(api_group)

        # --- Interaction ---
        query_group = QGroupBox("AI Interaction")
        query_layout = QVBoxLayout(query_group)
        self.model_combo = QComboBox()
        self.task_combo = QComboBox()
        self.task_combo.addItems(AI_CONTEXT_TASKS)
        self.prompt_edit = QTextEdit()
        self.send_btn = QPushButton(QIcon.fromTheme("mail-send"), "Send Query")
        self.response_browser = QTextBrowser()
        self.response_browser.setOpenExternalLinks(True)

        h_layout = QHBoxLayout()
        h_layout.addWidget(QLabel("Model:"))
        h_layout.addWidget(self.model_combo, 1)
        h_layout.addWidget(QLabel("Task:"))
        h_layout.addWidget(self.task_combo, 1)
        query_layout.addLayout(h_layout)
        query_layout.addWidget(QLabel("Your Prompt:"))
        query_layout.addWidget(self.prompt_edit)
        query_layout.addWidget(self.send_btn, 0, Qt.AlignmentFlag.AlignRight)
        query_layout.addWidget(QLabel("AI Response:"))
        query_layout.addWidget(self.response_browser, 1)
        main_layout.addWidget(query_group, 1)

        # --- Connections ---
        self.service_combo.currentTextChanged.connect(self._load_service_config)
        self.show_key_btn.toggled.connect(lambda c: self.api_key_edit.setEchoMode(QLineEdit.EchoMode.Normal if c else QLineEdit.EchoMode.Password))
        save_api_btn.clicked.connect(self._save_service_config)
        self.task_combo.currentTextChanged.connect(self._on_task_changed)
        self.send_btn.clicked.connect(self._send_query)

    def _load_service_config(self):
        service = self.service_combo.currentText()
        config = AI_SERVICES_CONFIG.get(service, {})
        
        self.model_combo.clear()
        self.model_combo.addItems(config.get("models", []))
        
        self.api_key_edit.setText(self.settings.value(f"AI/{config.get('key_name', '')}", "", type=str))
        
        has_url = "url_name" in config
        self.base_url_label.setVisible(has_url)
        self.base_url_edit.setVisible(has_url)
        if has_url:
            default_url = config.get("default_url", "")
            self.base_url_edit.setText(self.settings.value(f"AI/{config.get('url_name')}", default_url, type=str))
            self.base_url_edit.setPlaceholderText(default_url)

        self._update_send_button_state()

    def _save_service_config(self):
        service = self.service_combo.currentText()
        config = AI_SERVICES_CONFIG.get(service, {})
        if "key_name" in config:
            self.settings.setValue(f"AI/{config['key_name']}", self.api_key_edit.text())
        if "url_name" in config:
            self.settings.setValue(f"AI/{config['url_name']}", self.base_url_edit.text())
        
        QMessageBox.information(self, "Settings Saved", f"API configuration for {service} has been saved.")
        self._update_send_button_state()

    def _update_send_button_state(self):
        service = self.service_combo.currentText()
        config = AI_SERVICES_CONFIG.get(service, {})
        key_needed = "key_name" in config
        self.send_btn.setEnabled(not (key_needed and not self.api_key_edit.text()))

    def _on_task_changed(self, task: str):
        self.context_data = None
        prompts = {
            "Explain Parameter": "request_selected_parameter",
            "Interpret Sensitivity Results": "request_sa_results",
            "Interpret UQ Results": "request_uq_results",
        }
        if task in prompts:
            self.request_context_data.emit(prompts[task])
            self.response_browser.setHtml(f"<i>Requesting data for '{task}'... Please select the relevant item in its tab.</i>")

    def set_context_data(self, context_type: str, data: Any):
        self.context_data = {"type": context_type, "data": data}
        self.response_browser.setHtml(f"<i>Context received for '{context_type}'. Formulate your prompt.</i>")
        if context_type == "selected_parameter":
            self.prompt_edit.setText(f"Please explain the parameter '{data.get('name', 'N/A')}' and its impact on CO2 EOR.")

    def _send_query(self):
        if self.worker and self.worker.isRunning():
            QMessageBox.warning(self, "Busy", "An AI query is already in progress.")
            return

        service = self.service_combo.currentText()
        config = AI_SERVICES_CONFIG[service]
        prompt = self.prompt_edit.toPlainText().strip()
        if not prompt:
            QMessageBox.warning(self, "Empty Prompt", "Please enter a query.")
            return

        full_prompt = prompt
        if self.context_data:
            full_prompt += f"\n\n--- CONTEXT ---\n{self.context_data['type']}:\n{self.context_data['data']}"

        self.worker = AIQueryWorker(
            api_key=self.api_key_edit.text(),
            model_name=self.model_combo.currentText(),
            prompt=full_prompt,
            service_type=service.lower(),
            base_url=self.base_url_edit.text() if "url_name" in config else config.get("default_url")
        )
        self.worker.result_ready.connect(self.response_browser.setMarkdown)
        self.worker.error_occurred.connect(lambda e: self.response_browser.setHtml(f"<p style='color:red;'><b>Error:</b><br>{e}</p>"))
        self.worker.finished.connect(self._on_worker_finished)
        
        self.send_btn.setEnabled(False)
        self.response_browser.setHtml("<i>Sending query...</i>")
        self.worker.start()

    def _on_worker_finished(self):
        self.worker.deleteLater()
        self.worker = None
        self._update_send_button_state()