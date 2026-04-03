import logging
from typing import Optional, Any, Dict
from copy import deepcopy

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QLineEdit, QDialogButtonBox,
    QGroupBox, QListWidget, QListWidgetItem, QWidget, QHBoxLayout, QPushButton,
    QFileDialog, QLabel, QComboBox, QSpinBox
)
from PyQt6.QtCore import Qt, pyqtSignal, QEvent
import os

from ui.dialogs.ai_report_features_dialog import AIReportFeaturesDialog

logger = logging.getLogger(__name__)

class ReportConfigDialog(QDialog):
    """Dialog for configuring the sections and title of a generated report."""
    config_applied = pyqtSignal(dict)

    AVAILABLE_SECTIONS = {
        "project_summary": "Project Summary",
        "data_input_overview": "Data Input Overview",
        "mmp_analysis": "Miscibility Analysis",
        "eor_parameters_setup": "EOR Parameters Setup",
        "optimization_results": "Optimization Results",
        "sensitivity_analysis": "Sensitivity Analysis",
        "uq_analysis": "Uncertainty Quantification",
        "economic_assumptions": "Economic Assumptions",
        "conclusions": "Conclusions & Recommendations"
    }

    def __init__(self, current_config: Dict, project_name: str, data_availability: Dict[str, bool], parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setMinimumWidth(450)
        
        self.initial_config = deepcopy(current_config) or {}
        self.project_name = project_name

        if "report_title" not in self.initial_config:
            self.initial_config["report_title"] = f"{self.tr('CO2 EOR Analysis')}: {self.project_name}"
        if "sections" not in self.initial_config:
            self.initial_config["sections"] = {key: True for key in self.AVAILABLE_SECTIONS}
        if "output_path" not in self.initial_config:
            # Set default output path
            default_filename = f"CO2_EOR_Report_{project_name.replace(' ', '_')}.pdf"
            self.initial_config["output_path"] = os.path.join(os.path.expanduser("~"), default_filename)

        main_layout = QVBoxLayout(self)

        # Report title section
        title_label = QLabel(self.tr("Report Title:"))
        main_layout.addWidget(title_label)
        self.report_title_edit = QLineEdit()
        main_layout.addWidget(self.report_title_edit)

        # Output path section
        self.output_label = QLabel(self.tr("Output Path:"))
        main_layout.addWidget(self.output_label)
        
        output_layout = QHBoxLayout()
        self.output_path_edit = QLineEdit()
        self.output_path_edit.setText(self.initial_config.get("output_path", ""))
        output_layout.addWidget(self.output_path_edit)
        
        self.browse_button = QPushButton("...")
        self.browse_button.setFixedWidth(30)
        self.browse_button.clicked.connect(self._browse_output_path)
        output_layout.addWidget(self.browse_button)
        
        main_layout.addLayout(output_layout)

        # Formatting options section
        self.formatting_group = QGroupBox(self.tr("Formatting Options"))
        formatting_layout = QVBoxLayout(self.formatting_group)

        # Font family
        font_layout = QHBoxLayout()
        font_label = QLabel(self.tr("Font Family:"))
        font_layout.addWidget(font_label)
        self.font_combo = QComboBox()
        self.font_combo.addItems(["Times New Roman", "Arial", "Helvetica", "Calibri"])
        font_layout.addWidget(self.font_combo)
        formatting_layout.addLayout(font_layout)

        # Font size
        font_size_layout = QHBoxLayout()
        font_size_label = QLabel(self.tr("Font Size:"))
        font_size_layout.addWidget(font_size_label)
        self.font_size_spin = QSpinBox()
        self.font_size_spin.setRange(8, 24)
        self.font_size_spin.setValue(14)
        font_size_layout.addWidget(self.font_size_spin)
        formatting_layout.addLayout(font_size_layout)

        # Line spacing
        line_spacing_layout = QHBoxLayout()
        line_spacing_label = QLabel(self.tr("Line Spacing:"))
        line_spacing_layout.addWidget(line_spacing_label)
        self.line_spacing_combo = QComboBox()
        self.line_spacing_combo.addItems(["1.0 (Single)", "1.15", "1.5", "2.0 (Double)"])
        line_spacing_layout.addWidget(self.line_spacing_combo)
        formatting_layout.addLayout(line_spacing_layout)

        # Margins
        margins_label = QLabel(self.tr("Margins (mm):"))
        formatting_layout.addWidget(margins_label)
        
        margins_layout = QHBoxLayout()
        for margin_name in ["Top", "Right", "Bottom", "Left"]:
            margin_vlayout = QVBoxLayout()
            margin_label = QLabel(margin_name)
            margin_vlayout.addWidget(margin_label)
            margin_spin = QSpinBox()
            margin_spin.setRange(5, 50)
            margin_spin.setValue(20)
            margin_vlayout.addWidget(margin_spin)
            margins_layout.addLayout(margin_vlayout)
            setattr(self, f"margin_{margin_name.lower()}_spin", margin_spin)
        
        formatting_layout.addLayout(margins_layout)
        main_layout.addWidget(self.formatting_group)

        self.sections_group = QGroupBox()
        sections_layout = QVBoxLayout(self.sections_group)
        
        self.sections_list = QListWidget()
        for key, display_name in self.AVAILABLE_SECTIONS.items():
            item = QListWidgetItem()
            item.setData(Qt.ItemDataRole.UserRole, key)
            
            is_available = data_availability.get(key, True)
            if is_available:
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                is_checked = self.initial_config["sections"].get(key, True)
                item.setCheckState(Qt.CheckState.Checked if is_checked else Qt.CheckState.Unchecked)
            else:
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEnabled)
                item.setCheckState(Qt.CheckState.Unchecked)

            self.sections_list.addItem(item)
        sections_layout.addWidget(self.sections_list)

        select_buttons_layout = QHBoxLayout()
        self.select_all_btn = QPushButton()
        self.deselect_all_btn = QPushButton()
        select_buttons_layout.addWidget(self.select_all_btn)
        select_buttons_layout.addWidget(self.deselect_all_btn)
        select_buttons_layout.addStretch()
        sections_layout.addLayout(select_buttons_layout)
        
        main_layout.addWidget(self.sections_group)
        
        self.button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        main_layout.addWidget(self.button_box)
        self.button_box.accepted.connect(self.apply_and_accept)
        self.button_box.rejected.connect(self.reject)
        self.select_all_btn.clicked.connect(lambda: self._set_all_checks(Qt.CheckState.Checked))
        self.deselect_all_btn.clicked.connect(lambda: self._set_all_checks(Qt.CheckState.Unchecked))

        self.ai_features_btn = QPushButton(self.tr("Define AI Features..."))
        self.ai_features_btn.clicked.connect(self._open_ai_features_dialog)
        sections_layout.addWidget(self.ai_features_btn)

        self.ai_features = {}

        # Load initial formatting config
        self._load_formatting_config()

        self.retranslateUi()

    def _open_ai_features_dialog(self):
        summarizable_sections = {
            key: name for key, name in self.AVAILABLE_SECTIONS.items()
            if key in ["optimization_results", "sensitivity_analysis", "uq_analysis"]
        }
        dialog = AIReportFeaturesDialog(summarizable_sections, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.ai_features = dialog.get_selected_features()
            if self.ai_features:
                self.ai_features_btn.setStyleSheet("background-color: lightgreen")
            else:
                self.ai_features_btn.setStyleSheet("")

    def _load_formatting_config(self):
        """Load initial formatting configuration from initial_config"""
        format_options = self.initial_config.get('format_options', {})
        
        # Set font family
        font_family = format_options.get('font_family', 'Times New Roman')
        index = self.font_combo.findText(font_family)
        if index >= 0:
            self.font_combo.setCurrentIndex(index)
        
        # Set font size
        font_size = format_options.get('font_size', 14)
        self.font_size_spin.setValue(font_size)
        
        # Set line spacing
        line_spacing = format_options.get('line_spacing', 1.0)
        if line_spacing == 1.0:
            self.line_spacing_combo.setCurrentIndex(0)
        elif line_spacing == 1.15:
            self.line_spacing_combo.setCurrentIndex(1)
        elif line_spacing == 1.5:
            self.line_spacing_combo.setCurrentIndex(2)
        elif line_spacing == 2.0:
            self.line_spacing_combo.setCurrentIndex(3)
        
        # Set margins
        margins = format_options.get('margins', {'top': 20, 'right': 20, 'bottom': 20, 'left': 20})
        self.margin_top_spin.setValue(margins.get('top', 20))
        self.margin_right_spin.setValue(margins.get('right', 20))
        self.margin_bottom_spin.setValue(margins.get('bottom', 20))
        self.margin_left_spin.setValue(margins.get('left', 20))

    def retranslateUi(self):
        """Set or update the text for all translatable UI elements."""
        self.setWindowTitle(self.tr("Configure Report Content"))

        if not self.report_title_edit.text() or self.report_title_edit.text() == f"{self.tr('CO2 EOR Analysis')}: {self.project_name}":
             self.report_title_edit.setText(self.initial_config.get("report_title", f"{self.tr('CO2 EOR Analysis')}: {self.project_name}"))

        self.formatting_group.setTitle(self.tr("Formatting Options"))
        self.sections_group.setTitle(self.tr("Select Sections to Include"))
        
        self.output_label.setText(self.tr("Output Path:"))
        self.browse_button.setToolTip(self.tr("Browse for output location"))

        # Translate formatting options
        formatting_widgets = [
            (self.font_combo, ["Times New Roman", "Arial", "Helvetica", "Calibri"]),
            (self.line_spacing_combo, ["1.0 (Single)", "1.15", "1.5", "2.0 (Double)"])
        ]
        
        for widget, items in formatting_widgets:
            for i, item in enumerate(items):
                widget.setItemText(i, self.tr(item))

        for i in range(self.sections_list.count()):
            item = self.sections_list.item(i)
            key = item.data(Qt.ItemDataRole.UserRole)
            display_name = self.AVAILABLE_SECTIONS.get(key, key)
            item.setText(self.tr(display_name))

        self.select_all_btn.setText(self.tr("Select All"))
        self.deselect_all_btn.setText(self.tr("Deselect All"))
        self.ai_features_btn.setText(self.tr("Define AI Features..."))

    def changeEvent(self, event: QEvent):
        """Handle language change events to re-translate the UI."""
        if event.type() == QEvent.Type.LanguageChange:
            self.retranslateUi()
        super().changeEvent(event)

    def _set_all_checks(self, check_state: Qt.CheckState):
        for i in range(self.sections_list.count()):
            self.sections_list.item(i).setCheckState(check_state)

    def _browse_output_path(self):
        """Open file dialog to select output path."""
        default_path = self.output_path_edit.text() or os.path.expanduser("~")
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            self.tr("Select Report Output Location"),
            default_path,
            self.tr("PDF Files (*.pdf);;HTML Files (*.html);;All Files (*)")
        )
        if file_path:
            self.output_path_edit.setText(file_path)

    def get_config(self) -> Dict[str, Any]:
        """Constructs the configuration dictionary from the UI state."""
        output_path = self.output_path_edit.text().strip()
        if not output_path:
            default_filename = f"CO2_EOR_Report_{self.project_name.replace(' ', '_')}.pdf"
            output_path = os.path.join(os.path.expanduser("~"), default_filename)
        
        # Get line spacing value from combo box
        line_spacing_text = self.line_spacing_combo.currentText()
        line_spacing_map = {
            "1.0 (Single)": 1.0,
            "1.15": 1.15,
            "1.5": 1.5,
            "2.0 (Double)": 2.0
        }
        line_spacing = line_spacing_map.get(line_spacing_text, 1.0)
        
        config = {
            "report_title": self.report_title_edit.text().strip() or self.tr("Untitled Report"),
            "output_path": output_path,
            "ai_features": self.ai_features,
            "format_options": {
                "font_family": self.font_combo.currentText(),
                "font_size": self.font_size_spin.value(),
                "line_spacing": line_spacing,
                "margins": {
                    "top": self.margin_top_spin.value(),
                    "right": self.margin_right_spin.value(),
                    "bottom": self.margin_bottom_spin.value(),
                    "left": self.margin_left_spin.value()
                }
            }
        }
        sections = {
            self.sections_list.item(i).data(Qt.ItemDataRole.UserRole):
            self.sections_list.item(i).checkState() == Qt.CheckState.Checked
            for i in range(self.sections_list.count())
        }
        config["sections"] = sections
        return config

    def apply_and_accept(self):
        final_config = self.get_config()
        self.config_applied.emit(final_config)
        self.accept()

    @staticmethod
    def configure_report(current_config: Dict, proj_name: str, data_availability: Dict[str, bool], parent: QWidget) -> Optional[Dict]:
        dialog = ReportConfigDialog(current_config, proj_name, data_availability, parent)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            return dialog.get_config()
        return None