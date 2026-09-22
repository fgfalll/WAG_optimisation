import logging
from typing import Optional, Any, Dict
from copy import deepcopy

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QFormLayout, QLineEdit, QDialogButtonBox, QCheckBox, 
    QPushButton, QColorDialog, QLabel, QWidget, QHBoxLayout, QMessageBox
)
from PyQt6.QtGui import QColor, QPalette
# CORRECTED: Added 'Qt' to the import
from PyQt6.QtCore import pyqtSignal, Qt

logger = logging.getLogger(__name__)

class PlotCustomizationDialog(QDialog):
    """A dialog for customizing common plot appearance settings."""
    settings_applied = pyqtSignal(dict)

    CUSTOMIZABLE_ELEMENTS = [
        ("plot_title", "Plot Title", {"type": "text", "default": ""}),
        ("x_axis_label", "X-Axis Label", {"type": "text", "default": ""}),
        ("y_axis_label", "Y-Axis Label", {"type": "text", "default": ""}),
        ("legend_visible", "Show Legend", {"type": "bool", "default": True}),
        ("grid_visible", "Show Grid", {"type": "bool", "default": True}),
        ("background_color", "Background Color", {"type": "color", "default": "#FFFFFF"}),
    ]

    def __init__(self, current_settings: Optional[Dict[str, Any]] = None, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("Customize Plot Appearance")
        self.setMinimumWidth(400)
        # This line requires the 'Qt' import
        self.setWindowFlags(self.windowFlags() | Qt.WindowType.WindowSystemMenuHint | Qt.WindowType.WindowCloseButtonHint)

        self.initial_settings = deepcopy(current_settings) if current_settings else {}
        self.input_widgets: Dict[str, QWidget] = {}

        main_layout = QVBoxLayout(self)
        form_layout = QFormLayout()
        form_layout.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)

        for key, display_name, config in self.CUSTOMIZABLE_ELEMENTS:
            value = self.initial_settings.get(key, config["default"])
            widget = self._create_editor_widget(key, config, value)
            if widget:
                form_layout.addRow(QLabel(f"{display_name}:"), widget)

        main_layout.addLayout(form_layout)

        buttons = QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel | QDialogButtonBox.StandardButton.RestoreDefaults
        self.button_box = QDialogButtonBox(buttons)
        self.button_box.accepted.connect(self.apply_and_accept)
        self.button_box.rejected.connect(self.reject)
        self.button_box.button(QDialogButtonBox.StandardButton.RestoreDefaults).clicked.connect(self.restore_defaults)
        main_layout.addWidget(self.button_box)

    def _create_editor_widget(self, key: str, config: Dict, value: Any) -> Optional[QWidget]:
        widget_type = config["type"]
        if widget_type == "text":
            widget = QLineEdit(str(value))
        elif widget_type == "bool":
            widget = QCheckBox()
            widget.setChecked(bool(value))
        elif widget_type == "color":
            widget = self._create_color_picker(key, value)
        else:
            return None
        self.input_widgets[key] = widget
        return widget

    def _create_color_picker(self, key: str, value: Any) -> QWidget:
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        
        preview = QLabel()
        preview.setFixedSize(30, 20)
        preview.setAutoFillBackground(True)
        
        button = QPushButton("Choose...")
        initial_color = QColor(str(value))
        self._set_color_preview(preview, initial_color)
        
        self.input_widgets[key] = {"button": button, "preview": preview, "color": initial_color}
        
        button.clicked.connect(lambda: self._choose_color(key))
        
        layout.addWidget(preview)
        layout.addWidget(button)
        layout.addStretch()
        return container

    def _set_color_preview(self, preview_label: QLabel, color: QColor):
        palette = preview_label.palette()
        palette.setColor(QPalette.ColorRole.Window, color)
        preview_label.setPalette(palette)

    def _choose_color(self, settings_key: str):
        color_widgets = self.input_widgets[settings_key]
        if isinstance(color_widgets, dict):
            new_color = QColorDialog.getColor(color_widgets["color"], self, "Select Color")
            if new_color.isValid():
                color_widgets["color"] = new_color
                self._set_color_preview(color_widgets["preview"], new_color)

    def restore_defaults(self):
        for key, _, config in self.CUSTOMIZABLE_ELEMENTS:
            default = config["default"]
            widget = self.input_widgets.get(key)
            if not widget: continue

            if config["type"] == "text": widget.setText(str(default))
            elif config["type"] == "bool": widget.setChecked(bool(default))
            elif config["type"] == "color" and isinstance(widget, dict):
                default_qcolor = QColor(str(default))
                widget["color"] = default_qcolor
                self._set_color_preview(widget["preview"], default_qcolor)
        QMessageBox.information(self, "Defaults Restored", "Settings have been reset to defaults.")

    def get_current_settings(self) -> Dict[str, Any]:
        settings = {}
        for key, _, config in self.CUSTOMIZABLE_ELEMENTS:
            widget = self.input_widgets.get(key)
            if not widget: continue

            if config["type"] == "text": settings[key] = widget.text()
            elif config["type"] == "bool": settings[key] = widget.isChecked()
            elif config["type"] == "color" and isinstance(widget, dict):
                settings[key] = widget["color"].name()
        return settings

    def apply_and_accept(self):
        self.settings_applied.emit(self.get_current_settings())
        self.accept()

    @staticmethod
    def customize_plot(current_settings: Dict, parent: QWidget) -> Optional[Dict[str, Any]]:
        dialog = PlotCustomizationDialog(current_settings, parent)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            return dialog.get_current_settings()
        return None