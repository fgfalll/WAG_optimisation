import logging
from typing import Optional, Any, Dict

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QGroupBox, QLabel, QLineEdit,
    QComboBox, QTextEdit, QPushButton, QMessageBox, QScrollArea, QApplication, QInputDialog
)
from PyQt6.QtGui import QIcon, QTextCursor
from PyQt6.QtCore import Qt, QSettings, pyqtSignal, QEvent

from ui.workers.ai_query_worker import AIQueryWorker
from ui.workers.model_list_worker import ModelListWorker
from ui.widgets.chat_bubble import ChatBubble
from ui.widgets.context_bubble import ContextBubble
from ui.dialogs.task_editor_dialog import TaskEditorDialog
from utils.preferences_manager import PreferencesManager
from config_manager import ConfigManager

logger = logging.getLogger(__name__)

class AIAssistantWidget(QWidget):
    request_context_data = pyqtSignal(str)
    request_all_parameters = pyqtSignal()

    def __init__(self, app_settings: QSettings, pref_manager: PreferencesManager, config_manager: ConfigManager, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.settings = app_settings
        self.pref_manager = pref_manager
        self.config_manager = config_manager
        self.worker: Optional[AIQueryWorker] = None
        self.model_worker: Optional[ModelListWorker] = None
        self.context_data: Optional[Dict[str, Any]] = None
        self.all_models: list[str] = []
        self.current_ai_bubble: Optional[ChatBubble] = None
        self.current_ai_response: str = ""

        self.AI_SERVICES_CONFIG = self.config_manager.get_section("ui_config").get("ai_assistant", {}).get("services", {})
        self.DEFAULT_TASKS = self.config_manager.get_section("ui_config").get("ai_assistant", {}).get("default_tasks", {})

        self._init_ui()
        self._load_tasks()
        self._load_models_for_active_service()

        self.pref_manager.ai_preferences_changed.connect(self._load_models_for_active_service)

    def _init_ui(self):
        main_layout = QVBoxLayout(self)
        self.setLayout(main_layout)

        self.query_group = self._create_interaction_group()
        main_layout.addWidget(self.query_group, 1)

        self.retranslateUi()

    def _create_interaction_group(self) -> QGroupBox:
        query_group = QGroupBox()
        query_layout = QVBoxLayout(query_group)

        top_bar_layout = QHBoxLayout()
        self.active_service_label = QLabel()
        top_bar_layout.addWidget(self.active_service_label)
        top_bar_layout.addStretch()
        query_layout.addLayout(top_bar_layout)

        controls_layout = QHBoxLayout()
        self.model_label = QLabel()
        self.model_search_edit = QLineEdit()
        self.model_combo = QComboBox()
        self.refresh_models_btn = QPushButton()
        self.refresh_models_btn.setIcon(QIcon.fromTheme("view-refresh"))
        self.task_label = QLabel()
        self.task_combo = QComboBox()
        self.manage_tasks_btn = QPushButton("Manage Tasks...")

        model_layout = QVBoxLayout()
        model_layout.addWidget(self.model_label)
        model_layout.addWidget(self.model_search_edit)
        model_combo_layout = QHBoxLayout()
        model_combo_layout.addWidget(self.model_combo, 1)
        model_combo_layout.addWidget(self.refresh_models_btn)
        model_layout.addLayout(model_combo_layout)

        task_layout = QVBoxLayout()
        task_layout.addWidget(self.task_label)
        task_combo_layout = QHBoxLayout()
        task_combo_layout.addWidget(self.task_combo, 1)
        task_combo_layout.addWidget(self.manage_tasks_btn)
        task_layout.addLayout(task_combo_layout)
        task_layout.addStretch(1)

        controls_layout.addLayout(model_layout, 1)
        controls_layout.addLayout(task_layout, 1)
        query_layout.addLayout(controls_layout)

        self.chat_area = QScrollArea()
        self.chat_area.setWidgetResizable(True)
        self.chat_widget = QWidget()
        self.chat_layout = QVBoxLayout(self.chat_widget)
        self.chat_layout.addStretch()
        self.chat_area.setWidget(self.chat_widget)
        query_layout.addWidget(self.chat_area, 1)

        input_layout = QHBoxLayout()
        self.prompt_edit = QTextEdit()
        self.prompt_edit.setMinimumHeight(30)
        self.prompt_edit.setMaximumHeight(90)
        self.send_btn = QPushButton()
        self.send_btn.setIcon(QIcon.fromTheme("mail-send"))
        input_layout.addWidget(self.prompt_edit, 1)
        input_layout.addWidget(self.send_btn)
        query_layout.addLayout(input_layout)

        self.manage_tasks_btn.clicked.connect(self._open_task_editor)
        self.model_search_edit.textChanged.connect(self._filter_models)
        self.refresh_models_btn.clicked.connect(self._fetch_models_from_provider)
        self.task_combo.currentTextChanged.connect(self._on_task_changed)
        self.send_btn.clicked.connect(self._send_query)
        self.prompt_edit.installEventFilter(self)

        return query_group

    def retranslateUi(self):
        self.query_group.setTitle(self.tr("AI Assistant"))
        self.manage_tasks_btn.setText(self.tr("Manage Tasks..."))
        self.model_label.setText(self.tr("Model:"))
        self.model_search_edit.setPlaceholderText(self.tr("Search for model..."))
        self.refresh_models_btn.setToolTip(self.tr("Refresh model list from provider"))
        self.task_label.setText(self.tr("Task:"))
        self.prompt_edit.setPlaceholderText(self.tr("Ask a question or enter a prompt..."))
        self.send_btn.setToolTip(self.tr("Send Query"))

    def eventFilter(self, obj, event):
        if obj is self.prompt_edit and event.type() == QEvent.Type.KeyPress:
            if event.key() == Qt.Key.Key_Return and event.modifiers() == Qt.KeyboardModifier.NoModifier:
                self._send_query()
                return True
        return super().eventFilter(obj, event)

    def changeEvent(self, event: QEvent):
        if event.type() == QEvent.Type.LanguageChange:
            self.retranslateUi()
        super().changeEvent(event)

    def _clear_chat(self):
        while self.chat_layout.count() > 1:
            item = self.chat_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()

    def _filter_models(self, text: str):
        self.model_combo.clear()
        if not text:
            self.model_combo.addItems(self.all_models)
        else:
            filtered_models = [m for m in self.all_models if text.lower() in m.lower()]
            self.model_combo.addItems(filtered_models)

    def _load_models_for_active_service(self):
        ai_prefs = self.pref_manager.ai
        active_service = ai_prefs.active_service
        self.active_service_label.setText(f"Active Service: {active_service}")
        config = self.AI_SERVICES_CONFIG.get(active_service, {})
        self.all_models = config.get("models", [])
        self.model_search_edit.clear()
        self.model_combo.clear()
        self.model_combo.addItems(self.all_models)
        self.refresh_models_btn.setEnabled(config.get("model_endpoint") is not None)
        self._update_send_button_state()

    def _update_send_button_state(self):
        busy = (self.worker and self.worker.isRunning()) or (self.model_worker and self.model_worker.isRunning())
        self.send_btn.setEnabled(not busy)
        ai_prefs = self.pref_manager.ai
        active_service = ai_prefs.active_service
        api_key = ai_prefs.services.get(active_service, {}).get("api_key", "")
        if not api_key:
            self.send_btn.setEnabled(False)
        
    def _fetch_models_from_provider(self):
        if self.model_worker and self.model_worker.isRunning():
            return

        ai_prefs = self.pref_manager.ai
        service = ai_prefs.active_service
        service_config = ai_prefs.services.get(service, {})
        api_key = service_config.get("api_key")
        base_url = service_config.get("base_url")
        endpoint = AI_SERVICES_CONFIG.get(service, {}).get("model_endpoint")

        if not api_key:
            QMessageBox.warning(self, self.tr("API Key Required"), self.tr("API key is not set for the active service in Preferences."))
            return
            
        if not base_url or not endpoint:
            QMessageBox.warning(self, self.tr("Configuration Missing"), self.tr("Base URL or model endpoint is not configured for this service."))
            return

        self.model_worker = ModelListWorker(api_key=api_key, base_url=base_url, endpoint=endpoint, service_type=service.lower())
        self.model_worker.models_ready.connect(self._on_models_ready)
        self.model_worker.error_occurred.connect(self._on_model_fetch_error)
        self.model_worker.finished.connect(self._on_model_worker_finished)
        
        self.refresh_models_btn.setEnabled(False)
        self.model_combo.clear()
        self.model_combo.addItem(self.tr("Fetching models..."))
        self.model_worker.start()

    def _on_models_ready(self, models: list):
        self.all_models = models
        self.model_search_edit.clear()
        self.model_combo.clear()
        self.model_combo.addItems(self.all_models)
        QMessageBox.information(self, self.tr("Success"), self.tr("{count} models loaded.").format(count=len(models)))

    def _on_model_fetch_error(self, error_message: str):
        self.model_combo.clear()
        self.model_combo.addItem(self.tr("Error fetching models"))
        QMessageBox.critical(self, self.tr("Model Fetching Error"), error_message)
        self._load_models_for_active_service()

    def _on_model_worker_finished(self):
        self.refresh_models_btn.setEnabled(True)
        if self.model_worker:
            self.model_worker.deleteLater()
            self.model_worker = None
        self._update_send_button_state()

    def _open_task_editor(self):
        default_tasks = self.config_manager.get_section("ui_config").get("ai_assistant", {}).get("default_tasks", {})
        available_context_params = self.config_manager.get_section("ui_config").get("ai_assistant", {}).get("available_context_params", [])
        dialog = TaskEditorDialog(default_tasks, available_context_params, self)
        dialog.exec()
        self._load_tasks()

    def _load_tasks(self):
        self.task_combo.clear()
        for name, details in self.DEFAULT_TASKS.items():
            self.task_combo.addItem(name, userData=details)
        custom_tasks = self.settings.value("custom_tasks", {}, type=dict)
        for name, details in custom_tasks.items():
            self.task_combo.addItem(name, userData=details)

    def _on_task_changed(self, task_name: str):
        self.context_data = None
        self.prompt_edit.clear()
        
        if task_name == "Explain Parameter":
            self.request_all_parameters.emit()
        else:
            task_data = self.task_combo.currentData()
            if task_data:
                for context_param in task_data.get("context_params", []):
                    self.request_context_data.emit(context_param)

    def show_parameter_selection(self, params: Dict[str, list]):
        all_params = []
        for group, param_list in params.items():
            all_params.extend([f"{group}: {p}" for p in param_list])

        param, ok = QInputDialog.getItem(self, self.tr("Select Parameter"), self.tr("Parameter:"), all_params, 0, False)

        if ok and param:
            task_data = self.task_combo.currentData()
            system_prompt = task_data.get("prompt", "")
            prompt = f"{system_prompt}\n\nParameter: {param}"
            self.prompt_edit.setText(prompt)
            self._send_query_from_prompt(prompt, system_prompt)

    def set_context_data(self, context_type: str, data: Any):
        if self.context_data is None:
            self.context_data = {}
        self.context_data[context_type] = data
        context_bubble = ContextBubble(context_type, data)
        self.chat_layout.insertWidget(self.chat_layout.count() - 1, context_bubble)

    def _send_query(self):
        prompt = self.prompt_edit.toPlainText().strip()
        if not prompt:
            return
        
        task_data = self.task_combo.currentData()
        system_prompt = task_data.get("prompt", "") if task_data else ""
        self._send_query_from_prompt(prompt, system_prompt)

    def _send_query_from_prompt(self, prompt: str, system_prompt: str):
        if self.worker and self.worker.isRunning():
            return

        user_bubble = ChatBubble(prompt, is_user=True)
        user_bubble.request_copy.connect(lambda: self._handle_copy(user_bubble))
        user_bubble.request_edit.connect(lambda: self._handle_edit(user_bubble))
        
        user_bubble_wrapper = QWidget()
        user_bubble_layout = QHBoxLayout(user_bubble_wrapper)
        user_bubble_layout.addStretch()
        user_bubble_layout.addWidget(user_bubble)
        user_bubble_layout.setContentsMargins(0,0,0,0)
        self.chat_layout.insertWidget(self.chat_layout.count() - 1, user_bubble_wrapper)
        self.prompt_edit.clear()

        full_prompt = f"{system_prompt}\n\nUser Query: {prompt}"
        if self.context_data:
            context_str = "\n\n--- Context ---"
            for key, value in self.context_data.items():
                context_str += f"\nType: {key}\nData: {str(value)}"
            context_str += "\n--- End Context ---"
            full_prompt += context_str

        ai_prefs = self.pref_manager.ai
        service = ai_prefs.active_service
        service_config = ai_prefs.services.get(service, {})
        api_key = service_config.get("api_key")
        base_url = service_config.get("base_url")

        self.worker = AIQueryWorker(
            api_key=api_key,
            model_name=self.model_combo.currentText(),
            prompt=full_prompt,
            service_type=service.lower(),
            base_url=base_url
        )

        self.worker.partial_result_ready.connect(self._handle_partial_result)
        self.worker.error_occurred.connect(self._handle_error)
        self.worker.finished.connect(self._on_worker_finished)

        self.current_ai_response = ""
        self.current_ai_bubble = ChatBubble("", is_user=False)
        self.current_ai_bubble.request_copy.connect(lambda: self._handle_copy(self.current_ai_bubble))
        self.current_ai_bubble.request_regenerate.connect(lambda: self._handle_regenerate(self.current_ai_bubble, prompt, system_prompt))
        
        ai_bubble_wrapper = QWidget()
        ai_bubble_layout = QHBoxLayout(ai_bubble_wrapper)
        ai_bubble_layout.addWidget(self.current_ai_bubble)
        ai_bubble_layout.addStretch()
        ai_bubble_layout.setContentsMargins(0,0,0,0)
        self.chat_layout.insertWidget(self.chat_layout.count() - 1, ai_bubble_wrapper)

        self._update_send_button_state()
        self.worker.start()

    def _handle_copy(self, bubble: ChatBubble):
        QApplication.clipboard().setText(bubble.get_text())

    def _handle_edit(self, bubble: ChatBubble):
        self.prompt_edit.setText(bubble.get_text())

    def _handle_regenerate(self, ai_bubble: ChatBubble, prompt: str, system_prompt: str):
        for i in range(self.chat_layout.count()):
            item = self.chat_layout.itemAt(i)
            if item and item.widget() and item.widget().layout().indexOf(ai_bubble) != -1:
                wrapper = item.widget()
                wrapper.deleteLater()
                break
        self._send_query_from_prompt(prompt, system_prompt)

    def _handle_partial_result(self, text: str):
        if self.current_ai_bubble:
            self.current_ai_response += text
            self.current_ai_bubble.set_text(self.current_ai_response)
            self.chat_area.verticalScrollBar().setValue(self.chat_area.verticalScrollBar().maximum())

    def _handle_error(self, error_message: str):
        error_html = f"<p style='color:red;'><b>{self.tr('Error')}:</b><br>{error_message}</p>"
        error_bubble = ChatBubble(error_html, is_user=False)
        
        error_bubble_wrapper = QWidget()
        error_bubble_layout = QHBoxLayout(error_bubble_wrapper)
        error_bubble_layout.addWidget(error_bubble)
        error_bubble_layout.addStretch()
        error_bubble_layout.setContentsMargins(0,0,0,0)
        self.chat_layout.insertWidget(self.chat_layout.count() - 1, error_bubble_wrapper)

    def _on_worker_finished(self):
        if self.worker:
            self.worker.deleteLater()
            self.worker = None
        self.current_ai_bubble = None
        self.current_ai_response = ""
        self._update_send_button_state()