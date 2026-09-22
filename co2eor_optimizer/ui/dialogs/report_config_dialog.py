import logging
from typing import Optional, Any, Dict
from copy import deepcopy

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QFormLayout, QLineEdit, QDialogButtonBox,
    QGroupBox, QListWidget, QListWidgetItem, QWidget, QHBoxLayout, QPushButton
)
from PyQt6.QtCore import Qt, pyqtSignal

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

    def __init__(self, current_config: Dict, project_name: str, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("Configure Report Content")
        self.setMinimumWidth(450)
        
        self.initial_config = deepcopy(current_config) or {}
        # Ensure default structure exists
        if "report_title" not in self.initial_config:
            self.initial_config["report_title"] = f"CO2 EOR Analysis: {project_name}"
        if "sections" not in self.initial_config:
            self.initial_config["sections"] = {key: True for key in self.AVAILABLE_SECTIONS}

        main_layout = QVBoxLayout(self)

        self.report_title_edit = QLineEdit(self.initial_config["report_title"])
        main_layout.addWidget(self.report_title_edit)

        sections_group = QGroupBox("Select Sections to Include")
        sections_layout = QVBoxLayout(sections_group)
        
        self.sections_list = QListWidget()
        for key, display_name in self.AVAILABLE_SECTIONS.items():
            item = QListWidgetItem(display_name)
            item.setData(Qt.ItemDataRole.UserRole, key)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            is_checked = self.initial_config["sections"].get(key, True)
            item.setCheckState(Qt.CheckState.Checked if is_checked else Qt.CheckState.Unchecked)
            self.sections_list.addItem(item)
        sections_layout.addWidget(self.sections_list)

        select_buttons_layout = QHBoxLayout()
        select_all_btn = QPushButton("Select All")
        deselect_all_btn = QPushButton("Deselect All")
        select_buttons_layout.addWidget(select_all_btn)
        select_buttons_layout.addWidget(deselect_all_btn)
        select_buttons_layout.addStretch()
        sections_layout.addLayout(select_buttons_layout)
        
        main_layout.addWidget(sections_group)
        
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.apply_and_accept)
        buttons.rejected.connect(self.reject)
        main_layout.addWidget(buttons)

        select_all_btn.clicked.connect(lambda: self._set_all_checks(Qt.CheckState.Checked))
        deselect_all_btn.clicked.connect(lambda: self._set_all_checks(Qt.CheckState.Unchecked))

    def _set_all_checks(self, check_state: Qt.CheckState):
        for i in range(self.sections_list.count()):
            self.sections_list.item(i).setCheckState(check_state)

    def get_config(self) -> Dict[str, Any]:
        """Constructs the configuration dictionary from the UI state."""
        config = {"report_title": self.report_title_edit.text().strip() or "Untitled Report"}
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
    def configure_report(current_config: Dict, proj_name: str, parent: QWidget) -> Optional[Dict]:
        dialog = ReportConfigDialog(current_config, proj_name, parent)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            return dialog.get_config()
        return None