import json
from PyQt6.QtWidgets import (
    QDialog, QHBoxLayout, QVBoxLayout, QListWidget, QLineEdit, QTextEdit,
    QPushButton, QSplitter, QWidget, QGroupBox, QCheckBox, QMessageBox, QListWidgetItem
)
from PyQt6.QtCore import QSettings, Qt



class TaskEditorDialog(QDialog):
    def __init__(self, default_tasks, available_context_params, parent=None):
        super().__init__(parent)
        self.settings = QSettings()
        self.DEFAULT_TASKS = default_tasks
        self.AVAILABLE_CONTEXT_PARAMS = available_context_params
        self.setWindowTitle("Task Editor")
        self.setMinimumSize(800, 600)

        self._init_ui()
        self._load_tasks()

    def _init_ui(self):
        main_layout = QHBoxLayout(self)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)

        # Left side: Task list
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        self.task_list = QListWidget()
        self.task_list.itemSelectionChanged.connect(self._on_task_selected)
        left_layout.addWidget(self.task_list)
        splitter.addWidget(left_widget)

        # Right side: Editor
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        
        self.task_name_edit = QLineEdit()
        right_layout.addWidget(self.task_name_edit)

        self.system_prompt_edit = QTextEdit()
        right_layout.addWidget(self.system_prompt_edit)

        # Context parameters
        context_group = QGroupBox("Context Parameters")
        context_layout = QVBoxLayout(context_group)
        self.context_checkboxes = {}
        for param in self.AVAILABLE_CONTEXT_PARAMS:
            checkbox = QCheckBox(param)
            self.context_checkboxes[param] = checkbox
            context_layout.addWidget(checkbox)
        right_layout.addWidget(context_group)

        # Buttons
        button_layout = QHBoxLayout()
        self.new_btn = QPushButton("New Task")
        self.save_btn = QPushButton("Save Task")
        self.delete_btn = QPushButton("Delete Task")
        self.new_btn.clicked.connect(self._new_task)
        self.save_btn.clicked.connect(self._save_task)
        self.delete_btn.clicked.connect(self._delete_task)
        button_layout.addWidget(self.new_btn)
        button_layout.addWidget(self.save_btn)
        button_layout.addWidget(self.delete_btn)
        right_layout.addLayout(button_layout)

        right_widget.setLayout(right_layout)
        splitter.addWidget(right_widget)

    def _load_tasks(self):
        self.task_list.clear()
        # Load default tasks
        for name in self.DEFAULT_TASKS:
            item = QListWidgetItem(name)
            item.setData(Qt.ItemDataRole.UserRole, {"type": "default"})
            self.task_list.addItem(item)

        # Load custom tasks
        custom_tasks = self.settings.value("custom_tasks", {}, type=dict)
        for name in custom_tasks:
            item = QListWidgetItem(name)
            item.setData(Qt.ItemDataRole.UserRole, {"type": "custom"})
            self.task_list.addItem(item)

    def _on_task_selected(self):
        selected_items = self.task_list.selectedItems()
        if not selected_items:
            return

        item = selected_items[0]
        task_name = item.text()
        task_data = item.data(Qt.ItemDataRole.UserRole)
        is_custom = task_data["type"] == "custom"

        self.task_name_edit.setText(task_name)
        self.task_name_edit.setReadOnly(not is_custom)
        self.system_prompt_edit.setReadOnly(not is_custom)
        for checkbox in self.context_checkboxes.values():
            checkbox.setEnabled(is_custom)

        self.save_btn.setEnabled(is_custom)
        self.delete_btn.setEnabled(is_custom)

        if is_custom:
            custom_tasks = self.settings.value("custom_tasks", {}, type=dict)
            task_details = custom_tasks.get(task_name, {})
        else:
            task_details = self.DEFAULT_TASKS.get(task_name, {})

        self.system_prompt_edit.setText(task_details.get("prompt", ""))
        selected_params = task_details.get("context_params", [])
        for param, checkbox in self.context_checkboxes.items():
            checkbox.setChecked(param in selected_params)

    def _new_task(self):
        self.task_list.clearSelection()
        self.task_name_edit.clear()
        self.system_prompt_edit.clear()
        for checkbox in self.context_checkboxes.values():
            checkbox.setChecked(False)
        
        self.task_name_edit.setReadOnly(False)
        self.system_prompt_edit.setReadOnly(False)
        for checkbox in self.context_checkboxes.values():
            checkbox.setEnabled(True)
        self.save_btn.setEnabled(True)
        self.delete_btn.setEnabled(False)
        self.task_name_edit.setFocus()

    def _save_task(self):
        task_name = self.task_name_edit.text().strip()
        if not task_name:
            QMessageBox.warning(self, "Input Error", "Task name cannot be empty.")
            return

        custom_tasks = self.settings.value("custom_tasks", {}, type=dict)
        
        selected_params = [param for param, checkbox in self.context_checkboxes.items() if checkbox.isChecked()]
        
        task_details = {
            "prompt": self.system_prompt_edit.toPlainText(),
            "context_params": selected_params
        }

        custom_tasks[task_name] = task_details
        self.settings.setValue("custom_tasks", custom_tasks)
        self._load_tasks()
        QMessageBox.information(self, "Success", f"Task '{task_name}' saved.")

    def _delete_task(self):
        selected_items = self.task_list.selectedItems()
        if not selected_items:
            return

        task_name = selected_items[0].text()
        reply = QMessageBox.question(self, "Confirm Delete", f"Are you sure you want to delete the task '{task_name}'?")
        if reply == QMessageBox.StandardButton.Yes:
            custom_tasks = self.settings.value("custom_tasks", {}, type=dict)
            if task_name in custom_tasks:
                del custom_tasks[task_name]
                self.settings.setValue("custom_tasks", custom_tasks)
                self._load_tasks()
                self._new_task() # Clear the form
