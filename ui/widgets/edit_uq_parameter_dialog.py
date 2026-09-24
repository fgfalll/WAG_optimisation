from copy import deepcopy
from typing import Optional, Dict, List, Tuple, Any

from PyQt6.QtWidgets import QDialog, QFormLayout, QComboBox, QLabel, QDoubleSpinBox, QDialogButtonBox, QWidget
from PyQt6.QtCore import QEvent


SUPPORTED_DISTRIBUTIONS = ["Normal", "Uniform", "LogNormal", "Triangular"]
DIST_PARAMS_CONFIG = {
    "Normal": [("Mean (μ)", "mu"), ("Std Dev (σ)", "sigma")],
    "Uniform": [("Min", "lower"), ("Max", "upper")],
    "LogNormal": [("Log Mean (μ)", "log_mu"), ("Log Std Dev (σ)", "log_sigma")],
    "Triangular": [("Min (a)", "lower"), ("Mode (c)", "mode"), ("Max (b)", "upper")],
}


class EditUQParameterDialog(QDialog):
    """Dialog to add or edit a single uncertain parameter for UQ."""

    def __init__(
        self,
        all_params: List[Tuple[str, str]],
        existing_paths: List[str],
        supported_distributions: Optional[List[str]] = None,
        dist_params_config: Optional[Dict[str, Any]] = None,
        param_data: Optional[Dict] = None,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.param_data = deepcopy(param_data) if param_data else {}
        self.SUPPORTED_DISTRIBUTIONS = supported_distributions or SUPPORTED_DISTRIBUTIONS
        self.DIST_PARAMS_CONFIG = dist_params_config or DIST_PARAMS_CONFIG
        self._is_edit_mode = param_data is not None

        self._setup_ui(all_params, existing_paths)
        self._connect_signals()
        self.retranslateUi()

        if self.param_data:
            idx = self.param_combo.findData(self.param_data.get("path"))
            if idx != -1:
                self.param_combo.setCurrentIndex(idx)
            self.param_combo.setEnabled(False)
            self.dist_combo.setCurrentText(self.param_data.get("distribution", "Normal"))
            self._populate_dist_params(self.dist_combo.currentText())

    def _setup_ui(self, all_params, existing_paths):
        self.layout = QFormLayout(self)
        self.param_combo = QComboBox()
        self.param_combo.setEditable(True)
        for name, path in all_params:
            if path not in existing_paths or (
                self.param_data and self.param_data.get("path") == path
            ):
                self.param_combo.addItem(name, userData=path)

        self.dist_combo = QComboBox()
        self.dist_combo.addItems(self.SUPPORTED_DISTRIBUTIONS)

        self.param_label = QLabel()
        self.dist_label = QLabel()
        self.layout.addRow(self.param_label, self.param_combo)
        self.layout.addRow(self.dist_label, self.dist_combo)

        self.param_inputs: Dict[str, QDoubleSpinBox] = {}
        self.param_labels: Dict[str, QLabel] = {}
        self._params_widget = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self._params_widget.accepted.connect(self.accept)
        self._params_widget.rejected.connect(self.reject)
        self.layout.addRow(self._params_widget)

    def _connect_signals(self):
        self.dist_combo.currentTextChanged.connect(self._populate_dist_params)

    def retranslateUi(self):
        title = (
            self.tr("Edit Uncertain Parameter")
            if self._is_edit_mode
            else self.tr("Add Uncertain Parameter")
        )
        self.setWindowTitle(title)
        self.param_label.setText(self.tr("Parameter:"))
        self.dist_label.setText(self.tr("Distribution:"))

    def changeEvent(self, event: QEvent):
        if event.type() == QEvent.Type.LanguageChange:
            self.retranslateUi()
        super().changeEvent(event)

    def _populate_dist_params(self, dist_type: str):
        for w in self.param_inputs.values():
            self.layout.removeRow(w)
            w.deleteLater()
        for l in self.param_labels.values():
            l.deleteLater()
        self.param_inputs.clear()
        self.param_labels.clear()

        params_to_create = self.DIST_PARAMS_CONFIG.get(dist_type, [])
        current_params = self.param_data.get("params", [])

        insert_row = self.layout.rowCount() - 1
        for i, (name, _) in enumerate(params_to_create):
            label = QLabel(self.tr(name))
            widget = QDoubleSpinBox()
            widget.setDecimals(4)
            widget.setRange(-1e9, 1e9)
            if i < len(current_params):
                widget.setValue(current_params[i])
            self.param_inputs[name] = widget
            self.param_labels[name] = label
            self.layout.insertRow(insert_row, label, widget)

    def get_data(self) -> Optional[Dict]:
        path = self.param_combo.currentData()
        if path is None:
            return None
        parts = path.split(".")
        scope = parts[0]
        internal_name = parts[1] if len(parts) > 1 else ""
        return {
            "name": self.param_combo.currentText(),
            "path": path,
            "distribution": self.dist_combo.currentText(),
            "params": [w.value() for w in self.param_inputs.values()],
            "scope": scope,
            "internal_name": internal_name,
        }