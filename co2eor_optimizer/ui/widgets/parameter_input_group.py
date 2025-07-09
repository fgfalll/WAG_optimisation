import logging
from typing import Any, Optional, List, Tuple

from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QLabel, QLineEdit, QComboBox, QPushButton, 
    QSizePolicy, QMessageBox, QVBoxLayout, QFrame, QCheckBox
)
from PyQt6.QtCore import pyqtSignal, QTimer

logger = logging.getLogger(__name__)

class ParameterInputGroup(QWidget):
    """
    A reusable widget for a single parameter, now with multi-level feedback
    (error, warning, info) and debouncing for a smooth user experience.
    """
    finalValueChanged = pyqtSignal(object)

    def __init__(self, param_name: str, label_text: str, input_type: str, **kwargs):
        super().__init__(kwargs.get("parent"))
        self.param_name = param_name
        self.help_text = kwargs.get("help_text", "No help available for this parameter.")
        self.input_type = input_type.lower()

        self._debounce_timer = QTimer(self)
        self._debounce_timer.setSingleShot(True)
        # -- ENHANCEMENT: Increased debounce timer for a better user experience.
        self._debounce_timer.setInterval(500) 
        self._debounce_timer.timeout.connect(self._emit_debounced_value)
        
        self._main_layout = QVBoxLayout(self)
        self._main_layout.setContentsMargins(0, 0, 0, 0)
        self._main_layout.setSpacing(2)

        self.input_row_widget = QFrame()
        self.input_row_widget.setObjectName("InputWidgetFrame")
        self._input_row_layout = QHBoxLayout(self.input_row_widget)
        self._input_row_layout.setContentsMargins(2, 2, 2, 2)
        self._input_row_layout.setSpacing(5)

        self.label = QLabel(f"{label_text}:")
        self._input_row_layout.addWidget(self.label)

        self.input_widget = self._create_input_widget(kwargs)
        self.input_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._input_row_layout.addWidget(self.input_widget, 1)

        self.help_button = QPushButton("?")
        self.help_button.setFixedSize(22, 22)
        self.help_button.setToolTip("Show help for this parameter")
        self.help_button.clicked.connect(self._show_help)
        self._input_row_layout.addWidget(self.help_button)
        
        self.feedback_label = QLabel()
        self.feedback_label.setObjectName("FeedbackLabel")
        self.feedback_label.setVisible(False)
        self.feedback_label.setWordWrap(True)
        
        self._main_layout.addWidget(self.input_row_widget)
        self._main_layout.addWidget(self.feedback_label)

    def _create_input_widget(self, kwargs):
        default_value = kwargs.get("default_value")
        if self.input_type in ["lineedit", "doublespinbox", "spinbox"]:
            widget = QLineEdit()
            if "placeholder_text" in kwargs: widget.setPlaceholderText(kwargs["placeholder_text"])
            if default_value is not None:
                if isinstance(default_value, list): text_value = ", ".join(map(str, default_value))
                else: text_value = str(default_value)
                widget.setText(text_value.replace('.', ','))
            widget.textChanged.connect(self._start_debounce)
        elif self.input_type == "combobox":
            widget = QComboBox()
            items = kwargs.get("items", [])
            for item_data in items:
                if isinstance(item_data, tuple): widget.addItem(str(item_data[0]), userData=item_data[1])
                else: widget.addItem(str(item_data))
            if default_value is not None:
                idx = widget.findData(default_value);
                if idx == -1: idx = widget.findText(str(default_value))
                if idx != -1: widget.setCurrentIndex(idx)
            widget.currentIndexChanged.connect(self._emit_debounced_value)
        elif self.input_type == "checkbox":
            widget = QCheckBox()
            if default_value is not None:
                widget.setChecked(bool(default_value))
            widget.stateChanged.connect(self._start_debounce)
        else: raise ValueError(f"Unsupported input_type '{self.input_type}'")
        return widget

    def _start_debounce(self): self._debounce_timer.start()
    def _emit_debounced_value(self): self.finalValueChanged.emit(self.get_value())

    def _set_feedback(self, message: str, level: Optional[str]):
        if not level:
            self.feedback_label.setVisible(False)
            self.input_row_widget.setProperty("feedbackLevel", "none")
        else:
            self.feedback_label.setText(message)
            self.feedback_label.setVisible(True)
            self.input_row_widget.setProperty("feedbackLevel", level)
        
        self.input_row_widget.style().polish(self.input_row_widget)
        self.feedback_label.style().polish(self.feedback_label)

    def clear_error(self): self._set_feedback("", None)
    def show_error(self, message: str): self._set_feedback(message, "error")
    def show_warning(self, message: str): self._set_feedback(message, "warning")
    def show_info(self, message: str): self._set_feedback(message, "info")

    def set_checkable(self, checkable: bool):
        """Enable/disable the checkbox functionality for optional parameters"""
        if not hasattr(self, 'checkbox'):
            # Create checkbox if it doesn't exist
            self.checkbox = QCheckBox()
            self.checkbox.setChecked(True)
            self._input_row_layout.insertWidget(0, self.checkbox)
            self.checkbox.toggled.connect(self._on_checkbox_toggled)
        
        self.checkbox.setVisible(checkable)
        self._update_input_enabled()
        
    def is_checkable(self) -> bool:
        """Check if the widget has checkbox functionality enabled"""
        return hasattr(self, 'checkbox') and self.checkbox.isVisible()

    def _on_checkbox_toggled(self, checked):
        self._update_input_enabled()
        self._emit_debounced_value()

    def _update_input_enabled(self):
        enabled = not hasattr(self, 'checkbox') or self.checkbox.isChecked()
        self.label.setEnabled(enabled)
        self.input_widget.setEnabled(enabled)
        self.help_button.setEnabled(enabled)

    def is_checked(self) -> bool:
        """Check if the input widget is checked (for optional parameters)"""
        return not hasattr(self, 'checkbox') or self.checkbox.isChecked()

    def set_checked(self, checked: bool):
        """Set the checked state of the input widget"""
        if hasattr(self, 'checkbox'):
            self.checkbox.setChecked(checked)

    def _show_help(self):
        QMessageBox.information(self, f"Help: {self.label.text()[:-1]}", self.help_text)

    def get_value(self) -> Any:
        if self.input_type in ["lineedit", "doublespinbox", "spinbox"]:
            return self.input_widget.text().replace(',', '.')
        if self.input_type == "combobox":
            data = self.input_widget.currentData()
            return data if data is not None else self.input_widget.currentText()
        if self.input_type == "checkbox":
            return self.input_widget.isChecked()
        return None

    def set_value(self, value: Any):
        self.input_widget.blockSignals(True)
        try:
            if self.input_type in ["lineedit", "doublespinbox", "spinbox"]:
                text_value = str(value) if value is not None else ""
                self.input_widget.setText(text_value.replace('.', ','))
            elif self.input_type == "combobox":
                idx = self.input_widget.findData(value)
                if idx == -1: idx = self.input_widget.findText(str(value))
                if idx != -1: self.input_widget.setCurrentIndex(idx)
            elif self.input_type == "checkbox":
                self.input_widget.setChecked(bool(value))
        finally:
            self.input_widget.blockSignals(False)