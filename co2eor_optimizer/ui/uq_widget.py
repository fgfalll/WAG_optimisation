import logging
from typing import Optional, Any, Dict, List, Tuple
from copy import deepcopy
import pandas as pd

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QGroupBox, QLabel,
    QComboBox, QPushButton, QSplitter, QTableWidget, QTableWidgetItem,
    QHeaderView, QAbstractItemView, QMessageBox, QDialog, QFormLayout,
    QDoubleSpinBox, QSpinBox, QDialogButtonBox, QTextBrowser  # Added missing widgets
)
from PyQt6.QtGui import QIcon
from PyQt6.QtCore import pyqtSignal, Qt  # Added Qt namespace
from PyQt6.QtWebEngineWidgets import QWebEngineView

try:
    from .workers.uq_worker import UQWorker
    from co2eor_optimizer.analysis.uq_engine import UncertaintyQuantificationEngine
    from co2eor_optimizer.core.optimisation_engine import OptimizationEngine
except ImportError as e:
    logging.critical(f"UQWidget: Failed to import critical components: {e}.")
    UQWorker, UncertaintyQuantificationEngine, OptimizationEngine = None, object, object

logger = logging.getLogger(__name__)

SUPPORTED_DISTRIBUTIONS = ["Normal", "Uniform", "LogNormal", "Triangular"]
DIST_PARAMS_CONFIG = {
    "Normal": [("Mean (μ)", "mu"), ("Std Dev (σ)", "sigma")],
    "Uniform": [("Min", "lower"), ("Max", "upper")],
    "LogNormal": [("Log Mean (μ)", "log_mu"), ("Log Std Dev (σ)", "log_sigma")],
    "Triangular": [("Min (a)", "lower"), ("Mode (c)", "mode"), ("Max (b)", "upper")],
}

class EditUQParameterDialog(QDialog):
    """Dialog to add or edit a single uncertain parameter."""
    def __init__(self, all_params: List[Tuple[str, str]], existing_paths: List[str], 
                 param_data: Optional[Dict] = None, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.param_data = deepcopy(param_data) if param_data else {}
        self.setWindowTitle("Edit Uncertain Parameter" if param_data else "Add Uncertain Parameter")
        
        layout = QFormLayout(self)
        self.param_combo = QComboBox()
        self.param_combo.setEditable(True) # Make it searchable
        for name, path in all_params:
            if path not in existing_paths or (self.param_data and self.param_data.get("path") == path):
                self.param_combo.addItem(name, userData=path)
        
        self.dist_combo = QComboBox()
        self.dist_combo.addItems(SUPPORTED_DISTRIBUTIONS)
        
        layout.addRow("Parameter:", self.param_combo)
        layout.addRow("Distribution:", self.dist_combo)

        self.param_inputs: Dict[str, QDoubleSpinBox] = {}
        self.dist_combo.currentTextChanged.connect(lambda t: self._populate_dist_params(t, layout))
        
        if self.param_data:
            idx = self.param_combo.findData(self.param_data.get("path"))
            if idx != -1: self.param_combo.setCurrentIndex(idx)
            self.param_combo.setEnabled(False) # Cannot change parameter when editing
            self.dist_combo.setCurrentText(self.param_data.get("distribution"))
        
        self._populate_dist_params(self.dist_combo.currentText(), layout)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)

    def _populate_dist_params(self, dist_type: str, layout: QFormLayout):
        for w in self.param_inputs.values(): w.parent().deleteLater() # Clear old widgets
        self.param_inputs.clear()
        
        params_to_create = DIST_PARAMS_CONFIG.get(dist_type, [])
        current_param_values = self.param_data.get("params", [])
        for i, (name, _) in enumerate(params_to_create):
            widget = QDoubleSpinBox()
            widget.setDecimals(4)
            widget.setRange(-1e9, 1e9)
            if i < len(current_param_values):
                widget.setValue(current_param_values[i])
            self.param_inputs[name] = widget
            layout.addRow(name, widget)

    def get_data(self) -> Dict:
        path = self.param_combo.currentData()
        return {
            "name": self.param_combo.currentText(),
            "path": path,
            "distribution": self.dist_combo.currentText(),
            "params": [w.value() for w in self.param_inputs.values()],
            "scope": path.split('.')[0],
            "internal_name": path.split('.')[1],
        }

class UQWidget(QWidget):
    """Tab for configuring and running Uncertainty Quantification (UQ) analyses."""
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.engine: 'Optional[OptimizationEngine]' = None
        self.uq_engine: 'Optional[UncertaintyQuantificationEngine]' = None
        self.worker: 'Optional[UQWorker]' = None
        self.all_params: List[Tuple[str, str]] = []
        
        self._setup_ui()
        self._connect_signals()
        
        if not all([UQWorker, UncertaintyQuantificationEngine]):
            self.setEnabled(False)
            QMessageBox.critical(self, "Component Error", "Core components for UQ are missing. This tab is disabled.")

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        config_panel = self._create_config_panel()
        results_panel = self._create_results_panel()

        splitter.addWidget(config_panel)
        splitter.addWidget(results_panel)
        splitter.setSizes([500, 600])
        main_layout.addWidget(splitter)

    def _create_config_panel(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        
        params_group = QGroupBox("1. Uncertain Parameters")
        params_layout = QVBoxLayout(params_group)
        self.params_table = QTableWidget(0, 3)
        self.params_table.setHorizontalHeaderLabels(["Parameter", "Distribution", "Details"])
        self.params_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.params_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.params_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        params_layout.addWidget(self.params_table)
        
        btn_layout = QHBoxLayout()
        self.add_btn = QPushButton(QIcon.fromTheme("list-add"), "Add")
        self.edit_btn = QPushButton(QIcon.fromTheme("document-edit"), "Edit")
        self.remove_btn = QPushButton(QIcon.fromTheme("list-remove"), "Remove")
        btn_layout.addWidget(self.add_btn)
        btn_layout.addWidget(self.edit_btn)
        btn_layout.addWidget(self.remove_btn)
        btn_layout.addStretch()
        params_layout.addLayout(btn_layout)
        layout.addWidget(params_group)

        run_group = QGroupBox("2. Run Analysis")
        run_grid = QGridLayout(run_group)
        self.analysis_type_combo = QComboBox()
        self.analysis_type_combo.addItems(["Monte Carlo", "Polynomial Chaos Expansion (PCE)"])
        self.num_samples_spin = QSpinBox()
        self.num_samples_spin.setRange(10, 100_000)
        self.num_samples_spin.setValue(1000)
        self.pce_order_spin = QSpinBox()
        self.pce_order_spin.setRange(1, 5)
        self.pce_order_spin.setValue(2)
        self.run_button = QPushButton(QIcon.fromTheme("system-run"), "Run UQ Analysis")
        self.run_button.setEnabled(False)
        
        run_grid.addWidget(QLabel("Analysis Type:"), 0, 0)
        run_grid.addWidget(self.analysis_type_combo, 0, 1)
        run_grid.addWidget(QLabel("MC Samples:"), 1, 0)
        run_grid.addWidget(self.num_samples_spin, 1, 1)
        run_grid.addWidget(QLabel("PCE Order:"), 2, 0)
        run_grid.addWidget(self.pce_order_spin, 2, 1)
        run_grid.addWidget(self.run_button, 3, 0, 1, 2)
        layout.addWidget(run_group)
        layout.addStretch()
        
        return container

    def _create_results_panel(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        results_group = QGroupBox("UQ Results")
        results_layout = QVBoxLayout(results_group)
        self.plot_view = QWebEngineView()
        self.summary_browser = QTextBrowser()
        results_layout.addWidget(self.plot_view, 1)
        results_layout.addWidget(self.summary_browser)
        layout.addWidget(results_group)
        return container

    def _connect_signals(self):
        self.add_btn.clicked.connect(self._add_parameter)
        self.edit_btn.clicked.connect(self._edit_parameter)
        self.params_table.doubleClicked.connect(self._edit_parameter)
        self.remove_btn.clicked.connect(self._remove_parameter)
        self.run_button.clicked.connect(self._run_analysis)

    def update_engine(self, engine: 'Optional[OptimizationEngine]'):
        """Public method to provide the base engine, from which the UQ engine is derived."""
        self.engine = engine
        self.run_button.setEnabled(self.engine is not None)
        if self.engine:
            self.all_params = self.engine.get_configurable_parameters_for_uq()
            self._rebuild_uq_engine()
            logger.info("UQWidget engine instance updated.")
        else:
            self.all_params.clear()
            self.params_table.setRowCount(0)
            self.uq_engine = None
            logger.warning("UQWidget engine instance removed.")

    def _rebuild_uq_engine(self):
        """Re-initializes the UQ engine with the current UI configuration."""
        if not self.engine: return
        config = {"parameters": [self.params_table.item(r, 0).data(Qt.ItemDataRole.UserRole) for r in range(self.params_table.rowCount())]}
        self.uq_engine = UncertaintyQuantificationEngine(self.engine, config)
        logger.info("UQEngine instance rebuilt with UI parameters.")

    def _add_parameter(self):
        existing = [item.data(Qt.ItemDataRole.UserRole)["path"] for item in [self.params_table.item(r,0) for r in range(self.params_table.rowCount())]]
        dialog = EditUQParameterDialog(self.all_params, existing, parent=self)
        if dialog.exec():
            data = dialog.get_data()
            row = self.params_table.rowCount()
            self.params_table.insertRow(row)
            self._update_table_row(row, data)
            self._rebuild_uq_engine()

    def _edit_parameter(self):
        row = self.params_table.currentRow()
        if row < 0: return
        data = self.params_table.item(row, 0).data(Qt.ItemDataRole.UserRole)
        dialog = EditUQParameterDialog(self.all_params, [], data, self)
        if dialog.exec():
            new_data = dialog.get_data()
            self._update_table_row(row, new_data)
            self._rebuild_uq_engine()

    def _remove_parameter(self):
        row = self.params_table.currentRow()
        if row >= 0:
            self.params_table.removeRow(row)
            self._rebuild_uq_engine()

    def _update_table_row(self, row: int, data: Dict):
        item = QTableWidgetItem(data["name"])
        item.setData(Qt.ItemDataRole.UserRole, data)
        self.params_table.setItem(row, 0, item)
        self.params_table.setItem(row, 1, QTableWidgetItem(data["distribution"]))
        self.params_table.setItem(row, 2, QTableWidgetItem(", ".join(f"{p:.3g}" for p in data["params"])))

    def _run_analysis(self):
        if not self.uq_engine: return
        analysis_type = self.analysis_type_combo.currentText()
        method_name = "run_mc_analysis" if "Monte" in analysis_type else "run_pce_analysis"
        kwargs = {"num_samples": self.num_samples_spin.value()} if "Monte" in analysis_type else {"poly_order": self.pce_order_spin.value()}
        
        self.run_button.setEnabled(False)
        self.summary_browser.setText("<i>Running analysis...</i>")
        self.plot_view.setHtml("")
        
        self.worker = UQWorker(self.uq_engine, method_name, kwargs)
        self.worker.result_ready.connect(self._on_result)
        self.worker.error_occurred.connect(self._on_error)
        self.worker.finished.connect(self._on_worker_finished)
        self.worker.start()

    def _on_result(self, result: Any):
        self.summary_browser.clear()
        if isinstance(result, pd.DataFrame): # MC Results
            stats = result.describe().transpose()
            self.summary_browser.setHtml(f"<b>Monte Carlo Results:</b><br>{stats.to_html()}")
            if self.uq_engine:
                fig = self.uq_engine.plot_mc_results(result)
                self.plot_view.setHtml(fig.to_html(include_plotlyjs='cdn'))
        elif isinstance(result, dict): # PCE Results
            html = "<b>PCE Results:</b><ul>"
            for k, v in result.items():
                html += f"<li><b>{k.replace('_', ' ').title()}:</b> {v}</li>"
            html += "</ul>"
            self.summary_browser.setHtml(html)
            if self.uq_engine:
                fig = self.uq_engine.plot_pce_sobol_indices(result)
                self.plot_view.setHtml(fig.to_html(include_plotlyjs='cdn'))

    def _on_error(self, error_msg: str):
        self.summary_browser.setHtml(f"<p style='color:red;'><b>Error:</b><br>{error_msg}</p>")

    def _on_worker_finished(self):
        self.run_button.setEnabled(self.engine is not None)
        if self.worker:
            self.worker.deleteLater()
            self.worker = None