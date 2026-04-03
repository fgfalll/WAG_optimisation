import logging
from typing import Optional, Any, Dict, List, Tuple
from copy import deepcopy
import pandas as pd

from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QGroupBox,
    QLabel,
    QComboBox,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QAbstractItemView,
    QMessageBox,
    QDialog,
    QFormLayout,
    QDoubleSpinBox,
    QSpinBox,
    QDialogButtonBox,
    QTextBrowser,
)
from PyQt6.QtGui import QIcon
from PyQt6.QtCore import pyqtSignal, Qt, QEvent
from PyQt6.QtWebEngineWidgets import QWebEngineView

# --- Mock Imports for standalone execution ---
try:
    from ui.workers.uq_worker import UQWorker
    from analysis.uq_engine import UncertaintyQuantificationEngine
    from core.optimisation_engine import OptimizationEngine
except ImportError as e:
    logging.warning(f"Could not import project-specific modules: {e}. Using mock objects.")
    UQWorker, UncertaintyQuantificationEngine, OptimizationEngine = None, object, object

logger = logging.getLogger(__name__)

# --- Constants ---
SUPPORTED_DISTRIBUTIONS = ["Normal", "Uniform", "LogNormal", "Triangular"]
DIST_PARAMS_CONFIG = {
    # The first element of the tuple is the label, which will be translated.
    "Normal": [("Mean (μ)", "mu"), ("Std Dev (σ)", "sigma")],
    "Uniform": [("Min", "lower"), ("Max", "upper")],
    "LogNormal": [("Log Mean (μ)", "log_mu"), ("Log Std Dev (σ)", "log_sigma")],
    "Triangular": [("Min (a)", "lower"), ("Mode (c)", "mode"), ("Max (b)", "upper")],
}


# --- Dialog ---
class EditUQParameterDialog(QDialog):
    """Dialog to add or edit a single uncertain parameter."""

    def __init__(
        self,
        all_params: List[Tuple[str, str]],
        existing_paths: List[str],
        param_data: Optional[Dict] = None,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.param_data = deepcopy(param_data) if param_data else {}

        self._is_edit_mode = param_data is not None

        # Setup UI
        self._layout = QFormLayout(self)
        self.param_combo = QComboBox()
        self.param_combo.setEditable(True)
        for name, path in all_params:
            if path not in existing_paths or (
                self.param_data and self.param_data.get("path") == path
            ):
                self.param_combo.addItem(name, userData=path)

        self.dist_combo = QComboBox()
        self.dist_combo.addItems(SUPPORTED_DISTRIBUTIONS)

        self._param_label = QLabel()  # Store for retranslation
        self._dist_label = QLabel()  # Store for retranslation
        self._layout.addRow(self._param_label, self.param_combo)
        self._layout.addRow(self._dist_label, self.dist_combo)

        self.param_inputs: Dict[str, QDoubleSpinBox] = {}
        self._param_widgets_layout = QFormLayout()

        if self.param_data:
            idx = self.param_combo.findData(self.param_data.get("path"))
            if idx != -1:
                self.param_combo.setCurrentIndex(idx)
            self.param_combo.setEnabled(False)
            self.dist_combo.setCurrentText(self.param_data.get("distribution", "Normal"))

        self.dist_combo.currentTextChanged.connect(self._populate_dist_params)
        self._populate_dist_params(self.dist_combo.currentText())

        self._layout.addRow(self._param_widgets_layout)

        self.buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        self._layout.addRow(self.buttons)

        self.retranslateUi()  # Set initial text

    def retranslateUi(self):
        """Updates all translatable text in the dialog."""
        title = (
            self.tr("Edit Uncertain Parameter")
            if self._is_edit_mode
            else self.tr("Add Uncertain Parameter")
        )
        self.setWindowTitle(title)
        self._param_label.setText(self.tr("Parameter:"))
        self._dist_label.setText(self.tr("Distribution:"))
        # Re-populate to translate parameter names
        self._populate_dist_params(self.dist_combo.currentText())

    def changeEvent(self, event: QEvent):
        """Handle language change events."""
        if event.type() == QEvent.Type.LanguageChange:
            self.retranslateUi()
        super().changeEvent(event)

    def _populate_dist_params(self, dist_type: str):
        # Clear existing parameter input widgets
        while self._param_widgets_layout.rowCount() > 0:
            self._param_widgets_layout.removeRow(0)
        self.param_inputs.clear()

        params_to_create = DIST_PARAMS_CONFIG.get(dist_type, [])
        current_param_values = self.param_data.get("params", [])

        for i, (name, _) in enumerate(params_to_create):
            widget = QDoubleSpinBox()
            widget.setDecimals(4)
            widget.setRange(-1e9, 1e9)
            if (
                self.param_data
                and self.param_data.get("distribution") == dist_type
                and i < len(current_param_values)
            ):
                widget.setValue(current_param_values[i])
            self.param_inputs[name] = widget
            self._param_widgets_layout.addRow(self.tr(name), widget)

    def get_data(self) -> Optional[Dict]:
        path = self.param_combo.currentData()
        if not path:
            return None

        return {
            "name": self.param_combo.currentText(),
            "path": path,
            "distribution": self.dist_combo.currentText(),
            "params": [w.value() for w in self.param_inputs.values()],
            "scope": path.split(".")[0],
            "internal_name": path.split(".")[1],
        }


# --- Child Widgets for Separation of Concerns ---


class ParameterSetupWidget(QWidget):
    """Manages the UI for configuring uncertain parameters."""

    parameters_changed = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._all_configurable_params: List[Tuple[str, str]] = []
        self._setup_ui()
        self._connect_signals()
        self.retranslateUi()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.params_group = QGroupBox()
        params_layout = QVBoxLayout(self.params_group)

        self.params_table = QTableWidget(0, 3)
        self.params_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.params_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.params_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        params_layout.addWidget(self.params_table)

        btn_layout = QHBoxLayout()
        self.add_btn = QPushButton(QIcon.fromTheme("list-add"))
        self.edit_btn = QPushButton(QIcon.fromTheme("document-edit"))
        self.remove_btn = QPushButton(QIcon.fromTheme("list-remove"))
        btn_layout.addWidget(self.add_btn)
        btn_layout.addWidget(self.edit_btn)
        btn_layout.addWidget(self.remove_btn)
        btn_layout.addStretch()
        params_layout.addLayout(btn_layout)

        layout.addWidget(self.params_group)

    def retranslateUi(self):
        """Updates all translatable text in the widget."""
        self.params_group.setTitle(self.tr("1. Uncertain Parameters"))
        self.params_table.setHorizontalHeaderLabels(
            [self.tr("Parameter"), self.tr("Distribution"), self.tr("Details")]
        )
        self.add_btn.setText(self.tr("Add"))
        self.edit_btn.setText(self.tr("Edit"))
        self.remove_btn.setText(self.tr("Remove"))

    def changeEvent(self, event: QEvent):
        """Handle language change events."""
        if event.type() == QEvent.Type.LanguageChange:
            self.retranslateUi()
        super().changeEvent(event)

    def _connect_signals(self):
        self.add_btn.clicked.connect(self._add_parameter)
        self.edit_btn.clicked.connect(self._edit_parameter)
        self.params_table.doubleClicked.connect(self._edit_parameter)
        self.remove_btn.clicked.connect(self._remove_parameter)

    def set_all_configurable_parameters(self, params: List[Tuple[str, str]]):
        self._all_configurable_params = params
        self.params_table.setRowCount(0)

    def get_parameters_config(self) -> List[Dict]:
        """Returns the configuration for all defined uncertain parameters."""
        return [
            self.params_table.item(r, 0).data(Qt.ItemDataRole.UserRole)
            for r in range(self.params_table.rowCount())
        ]

    def _add_parameter(self):
        existing_paths = [cfg["path"] for cfg in self.get_parameters_config()]
        dialog = EditUQParameterDialog(self._all_configurable_params, existing_paths, parent=self)
        if dialog.exec() and (data := dialog.get_data()):
            row = self.params_table.rowCount()
            self.params_table.insertRow(row)
            self._update_table_row(row, data)
            self.parameters_changed.emit()
        elif dialog.result() == QDialog.DialogCode.Accepted:
            QMessageBox.warning(
                self, self.tr("Invalid Parameter"), self.tr("The selected parameter is not valid.")
            )

    def _edit_parameter(self):
        row = self.params_table.currentRow()
        if row < 0:
            return

        data = self.params_table.item(row, 0).data(Qt.ItemDataRole.UserRole)
        dialog = EditUQParameterDialog(self._all_configurable_params, [], data, self)
        if dialog.exec() and (new_data := dialog.get_data()):
            self._update_table_row(row, new_data)
            self.parameters_changed.emit()

    def _remove_parameter(self):
        row = self.params_table.currentRow()
        if row >= 0:
            self.params_table.removeRow(row)
            self.parameters_changed.emit()

    def _update_table_row(self, row: int, data: Dict):
        item = QTableWidgetItem(data["name"])
        item.setData(Qt.ItemDataRole.UserRole, data)
        self.params_table.setItem(row, 0, item)
        self.params_table.setItem(row, 1, QTableWidgetItem(data["distribution"]))
        params_str = ", ".join(f"{p:.3g}" for p in data.get("params", []))
        self.params_table.setItem(row, 2, QTableWidgetItem(params_str))


class AnalysisRunWidget(QWidget):
    """Manages the UI for running a UQ analysis."""

    analysis_requested = pyqtSignal(dict)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._setup_ui()
        self._connect_signals()
        self.retranslateUi()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.run_group = QGroupBox()
        run_grid = QGridLayout(self.run_group)

        self.analysis_type_combo = QComboBox()

        self.num_samples_spin = QSpinBox()
        self.num_samples_spin.setRange(10, 100_000)
        self.num_samples_spin.setValue(1000)

        self.pce_order_spin = QSpinBox()
        self.pce_order_spin.setRange(1, 5)
        self.pce_order_spin.setValue(2)

        self.run_button = QPushButton(QIcon.fromTheme("system-run"))
        self.run_button.setEnabled(False)

        # Store labels to re-translate them
        self.analysis_type_label = QLabel()
        self.mc_samples_label = QLabel()
        self.pce_order_label = QLabel()

        run_grid.addWidget(self.analysis_type_label, 0, 0)
        run_grid.addWidget(self.analysis_type_combo, 0, 1)
        run_grid.addWidget(self.mc_samples_label, 1, 0)
        run_grid.addWidget(self.num_samples_spin, 1, 1)
        run_grid.addWidget(self.pce_order_label, 2, 0)
        run_grid.addWidget(self.pce_order_spin, 2, 1)
        run_grid.addWidget(self.run_button, 3, 0, 1, 2)

        layout.addWidget(self.run_group)
        layout.addStretch()

    def retranslateUi(self):
        """Updates all translatable text in the widget."""
        self.run_group.setTitle(self.tr("2. Run Analysis"))

        # Save current text to restore selection after re-populating
        current_text = self.analysis_type_combo.currentText()
        self.analysis_type_combo.clear()
        self.analysis_type_combo.addItems(
            [self.tr("Monte Carlo"), self.tr("Polynomial Chaos Expansion (PCE)")]
        )
        self.analysis_type_combo.setCurrentText(current_text)

        self.run_button.setText(self.tr("Run UQ Analysis"))
        self.analysis_type_label.setText(self.tr("Analysis Type:"))
        self.mc_samples_label.setText(self.tr("MC Samples:"))
        self.pce_order_label.setText(self.tr("PCE Order:"))
        # Re-apply visibility based on current selection
        self._toggle_options(self.analysis_type_combo.currentText())

    def changeEvent(self, event: QEvent):
        """Handle language change events."""
        if event.type() == QEvent.Type.LanguageChange:
            self.retranslateUi()
        super().changeEvent(event)

    def _connect_signals(self):
        self.run_button.clicked.connect(self._on_run_clicked)
        self.analysis_type_combo.currentTextChanged.connect(self._toggle_options)

    def _toggle_options(self, analysis_type_text: str):
        # Compare with translated string to determine visibility
        is_mc = self.tr("Monte Carlo") in analysis_type_text
        self.num_samples_spin.setVisible(is_mc)
        self.mc_samples_label.setVisible(is_mc)
        self.pce_order_spin.setVisible(not is_mc)
        self.pce_order_label.setVisible(not is_mc)

    def _on_run_clicked(self):
        analysis_type = self.analysis_type_combo.currentText()
        config = {"type": analysis_type}
        # Use translated strings for comparison
        if analysis_type == self.tr("Monte Carlo"):
            config["method"] = "run_mc_analysis"
            config["kwargs"] = {"num_samples": self.num_samples_spin.value()}
        else:  # PCE
            config["method"] = "run_pce_analysis"
            config["kwargs"] = {"poly_order": self.pce_order_spin.value()}
        self.analysis_requested.emit(config)

    def set_run_enabled(self, enabled: bool):
        self.run_button.setEnabled(enabled)


class ResultsDisplayWidget(QWidget):
    """Passive widget to display plots and summary text."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._setup_ui()
        self.retranslateUi()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.results_group = QGroupBox()
        results_layout = QVBoxLayout(self.results_group)

        self.plot_view = QWebEngineView()
        self.summary_browser = QTextBrowser()

        results_splitter = QSplitter(Qt.Orientation.Vertical)
        results_splitter.addWidget(self.plot_view)
        results_splitter.addWidget(self.summary_browser)
        # Use stretch factors for better resizing behavior
        results_splitter.setStretchFactor(0, 1)
        results_splitter.setStretchFactor(1, 0)
        results_splitter.setSizes([350, 150])  # Initial sizes only

        results_layout.addWidget(results_splitter)
        layout.addWidget(self.results_group)

    def retranslateUi(self):
        """Updates all translatable text in the widget."""
        self.results_group.setTitle(self.tr("UQ Results"))

    def changeEvent(self, event: QEvent):
        """Handle language change events."""
        if event.type() == QEvent.Type.LanguageChange:
            self.retranslateUi()
        super().changeEvent(event)

    def show_running(self):
        self.summary_browser.setText(self.tr("<i>Running analysis...</i>"))
        self.plot_view.setHtml("")

    def display_error(self, error_msg: str):
        error_html = self.tr("<p style='color:red;'><b>Error:</b><br>{error_msg}</p>").format(
            error_msg=error_msg
        )
        self.summary_browser.setHtml(error_html)

    def display_results(self, result: Any, uq_engine: UncertaintyQuantificationEngine):
        self.summary_browser.clear()
        if isinstance(result, pd.DataFrame):  # MC Results
            stats = result.describe().transpose()
            # Combine translated string with non-translated HTML data
            html_content = self.tr("<b>Monte Carlo Results:</b><br>") + stats.to_html()
            self.summary_browser.setHtml(html_content)
            if uq_engine:
                objective_col = result.columns[-1]
                fig = uq_engine.plot_mc_results(result, objective_col)
                if fig:
                    self.plot_view.setHtml(fig.to_html(include_plotlyjs="cdn"))

        elif isinstance(result, dict):  # PCE Results
            html = self.tr("<b>PCE Results:</b>") + "<ul>"
            for k, v in result.items():
                # The result keys (e.g., "sobol_total") are not user-facing strings
                # suitable for direct translation. We format them for readability.
                key_display = k.replace("_", " ").title()
                if isinstance(v, dict):
                    html += f"<li><b>{key_display}:</b><ul>"
                    for sk, sv in v.items():
                        html += f"<li>{sk}: {sv:.4f}</li>"
                    html += "</ul></li>"
                else:
                    html += f"<li><b>{key_display}:</b> {v}</li>"
            html += "</ul>"
            self.summary_browser.setHtml(html)
            if uq_engine:
                fig = uq_engine.plot_pce_sobol_indices(result)
                if fig:
                    self.plot_view.setHtml(fig.to_html(include_plotlyjs="cdn"))


# --- Main UQWidget (Controller) ---


class UQWidget(QWidget):
    """
    Main tab for configuring and running Uncertainty Quantification (UQ) analyses.
    Acts as a controller, coordinating child widgets and the backend engine.
    """

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.engine: "Optional[OptimizationEngine]" = None
        self.uq_engine: "Optional[UncertaintyQuantificationEngine]" = None
        self.worker: "Optional[UQWorker]" = None

        self._setup_ui()
        self._connect_signals()

        if not all([UQWorker, UncertaintyQuantificationEngine, OptimizationEngine]):
            self.setEnabled(False)
            QMessageBox.critical(
                self,
                self.tr("Component Error"),
                self.tr("Core components for UQ are missing. This tab is disabled."),
            )

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        splitter = QSplitter(Qt.Orientation.Horizontal)

        self.config_panel = QWidget()
        config_layout = QVBoxLayout(self.config_panel)
        self.param_setup_widget = ParameterSetupWidget()
        self.run_widget = AnalysisRunWidget()
        config_layout.addWidget(self.param_setup_widget)
        config_layout.addWidget(self.run_widget)

        self.results_widget = ResultsDisplayWidget()

        splitter.addWidget(self.config_panel)
        self.config_panel.setMinimumWidth(350)
        self.config_panel.setMaximumWidth(500)
        splitter.addWidget(self.results_widget)
        self.results_widget.setMinimumWidth(400)
        # Use stretch factors for better resizing behavior
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([400, 700])  # Initial sizes only
        main_layout.addWidget(splitter)

    def _connect_signals(self):
        self.param_setup_widget.parameters_changed.connect(self._rebuild_uq_engine)
        self.run_widget.analysis_requested.connect(self._run_analysis)

    def update_engine(self, engine: "Optional[OptimizationEngine]"):
        """Public method to provide the base engine, from which the UQ engine is derived."""
        self.engine = engine
        self.run_widget.set_run_enabled(self.engine is not None)

        if self.engine:
            all_params = self.engine.get_configurable_parameters_for_uq()
            self.param_setup_widget.set_all_configurable_parameters(all_params)
            self._rebuild_uq_engine()
            logger.info("UQWidget engine instance updated.")
        else:
            self.param_setup_widget.set_all_configurable_parameters([])
            self.uq_engine = None
            logger.warning("UQWidget engine instance removed.")

    def _rebuild_uq_engine(self):
        """Re-initializes the UQ engine with the current UI configuration."""
        if not self.engine:
            return

        config = {"parameters": self.param_setup_widget.get_parameters_config()}
        try:
            self.uq_engine = UncertaintyQuantificationEngine(self.engine, config)
            logger.info("UQEngine instance rebuilt with UI parameters.")
        except Exception as e:
            self.uq_engine = None
            logger.error(f"Failed to rebuild UQ engine: {e}")
            self.results_widget.display_error(f"Failed to configure UQ Engine: {e}")

    def _run_analysis(self, config: Dict):
        if not self.uq_engine:
            QMessageBox.critical(
                self,
                self.tr("Error"),
                self.tr("UQ Engine is not initialized. Configure parameters first."),
            )
            return

        self.run_widget.set_run_enabled(False)
        self.results_widget.show_running()

        self.worker = UQWorker(self.uq_engine, config["method"], config["kwargs"])
        self.worker.result_ready.connect(self._on_result)
        self.worker.error_occurred.connect(self.results_widget.display_error)
        self.worker.finished.connect(self._on_worker_finished)
        self.worker.start()

    def _on_result(self, result: Any):
        self.results_widget.display_results(result, self.uq_engine)

    def _on_worker_finished(self):
        self.run_widget.set_run_enabled(self.engine is not None)
        if self.worker:
            self.worker.deleteLater()
            self.worker = None
