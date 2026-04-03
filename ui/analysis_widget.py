from __future__ import annotations
import logging
from typing import Optional, Any, Dict, List, Tuple
from copy import deepcopy
import numpy as np
import pandas as pd

from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QGroupBox,
    QLabel,
    QLineEdit,
    QComboBox,
    QPushButton,
    QSplitter,
    QTabWidget,
    QTreeView,
    QAbstractItemView,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QMessageBox,
    QSpinBox,
    QDialog,
    QFormLayout,
    QDoubleSpinBox,
    QDialogButtonBox,
    QTextBrowser,
)
from PyQt6.QtGui import QIcon, QStandardItemModel, QStandardItem
from PyQt6.QtCore import pyqtSignal, Qt, QEvent

# Conditional Imports for Core Components and Plotting
from ui.sensitivity_widget import RedesignedSensitivityWidget
from ui.widgets.parameter_tree_view import ParameterSelectionTreeView

SensitivityAnalysisWorker: Optional[Any] = None
UQWorker: Optional[Any] = None
SensitivityAnalyzer: Optional[Any] = None
UncertaintyQuantificationEngine: Optional[Any] = None
OptimizationEngine: Optional[Any] = None
QWebEngineView: Optional[Any] = None
DeclineCurveAnalyzer: Optional[Any] = None
DCAResult: Optional[Any] = None

try:
    from ui.workers.sensitivity_analysis_worker import SensitivityAnalysisWorker
    from ui.workers.uq_worker import UQWorker
    from ui.workers.dca_worker import DCAWorker
    from analysis.sensitivity_analyzer import SensitivityAnalyzer
    from analysis.uq_engine import UncertaintyQuantificationEngine
    from core.optimisation_engine import OptimizationEngine
    from analysis.decline_curve_analysis import DeclineCurveAnalyzer, DCAResult
except ImportError as e:
    logging.critical(f"AnalysisWidget: Failed to import core optimizer components: {e}.")

try:
    from PyQt6.QtWebEngineWidgets import QWebEngineView as ImportedQWebEngineView

    QWebEngineView = ImportedQWebEngineView
except ImportError as e:
    logging.critical(
        f"AnalysisWidget: Failed to import PyQt6-WebEngineWidgets: {e}. Plotting features will be disabled."
    )

logger = logging.getLogger(__name__)


class EditUQParameterDialog(QDialog):
    """Dialog to add or edit a single uncertain parameter for UQ."""

    def __init__(
        self,
        all_params: List[Tuple[str, str]],
        existing_paths: List[str],
        supported_distributions: List[str],
        dist_params_config: Dict[str, Any],
        param_data: Optional[Dict] = None,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.param_data = deepcopy(param_data) if param_data else {}
        self.SUPPORTED_DISTRIBUTIONS = supported_distributions
        self.DIST_PARAMS_CONFIG = dist_params_config

        self._setup_ui(all_params, existing_paths)
        self._connect_signals()
        self.retranslateUi()

        if self.param_data:
            idx = self.param_combo.findData(self.param_data.get("path"))
            if idx != -1:
                self.param_combo.setCurrentIndex(idx)
            self.param_combo.setEnabled(False)
            self.dist_combo.setCurrentText(self.param_data.get("distribution"))

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

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        self.layout.addRow(buttons)

    def _connect_signals(self):
        self.dist_combo.currentTextChanged.connect(self._populate_dist_params)

    def retranslateUi(self):
        title = (
            self.tr("Edit Uncertain Parameter")
            if self.param_data
            else self.tr("Add Uncertain Parameter")
        )
        self.setWindowTitle(title)
        self.param_label.setText(self.tr("Parameter:"))
        self.dist_label.setText(self.tr("Distribution:"))
        # Re-translate dynamically added labels
        for name, label in self.param_labels.items():
            label.setText(self.tr(name))

    def changeEvent(self, event: QEvent):
        if event.type() == QEvent.Type.LanguageChange:
            self.retranslateUi()
        super().changeEvent(event)

    def _populate_dist_params(self, dist_type: str):
        for w in self.param_inputs.values():
            w.parent().deleteLater()
        for l in self.param_labels.values():
            l.parent().deleteLater()
        self.param_inputs.clear()
        self.param_labels.clear()

        params_to_create = self.DIST_PARAMS_CONFIG.get(dist_type, [])
        for i, (name, _) in enumerate(params_to_create):
            label = QLabel(self.tr(name))
            widget = QDoubleSpinBox()
            widget.setDecimals(4)
            widget.setRange(-1e9, 1e9)
            if (
                self.param_data
                and "params" in self.param_data
                and i < len(self.param_data["params"])
            ):
                widget.setValue(self.param_data["params"][i])
            self.param_inputs[name] = widget
            self.param_labels[name] = label
            self.layout.insertRow(self.layout.rowCount() - 1, label, widget)

    def get_data(self) -> Optional[Dict]:
        path = self.param_combo.currentData()
        if path is None:
            return None
        parts = path.split(".")
        scope = parts[0]
        internal_name = parts[1] if len(parts) > 1 else ""  # Handle cases where path has no dot
        return {
            "name": self.param_combo.currentText(),
            "path": path,
            "distribution": self.dist_combo.currentText(),
            "params": [w.value() for w in self.param_inputs.values()],
            "scope": scope,
            "internal_name": internal_name,
        }


# --- Main Consolidated Widget ---


class AnalysisWidget(QWidget):
    """A consolidated widget for Sensitivity and Uncertainty Quantification analyses."""

    help_requested = pyqtSignal(str)

    def __init__(self, config_manager: "ConfigManager", parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.analyzer: Optional[SensitivityAnalyzer] = None
        self.engine: Optional[OptimizationEngine] = None
        self.uq_engine: Optional[UncertaintyQuantificationEngine] = None

        self.sa_worker: Optional[SensitivityAnalysisWorker] = None
        self.uq_worker: Optional[UQWorker] = None
        self.dca_worker: Optional[DCAWorker] = None
        self.dca_results: Optional[DCAResult] = None
        self.dca_analyzer: Optional[DeclineCurveAnalyzer] = None
        if DeclineCurveAnalyzer:
            self.dca_analyzer = DeclineCurveAnalyzer()

        self.uq_all_params: List[Tuple[str, str]] = []

        self.SUPPORTED_DISTRIBUTIONS = (
            config_manager.get_section("ui_config")
            .get("analysis", {})
            .get("supported_distributions", [])
        )
        self.DIST_PARAMS_CONFIG = (
            config_manager.get_section("ui_config")
            .get("analysis", {})
            .get("distribution_parameters", {})
        )

        self._setup_ui()
        self.retranslateUi()
        self._connect_signals()

        if not all(
            [
                SensitivityAnalysisWorker,
                UQWorker,
                SensitivityAnalyzer,
                UncertaintyQuantificationEngine,
                DeclineCurveAnalyzer,
            ]
        ):
            self.setEnabled(False)
            QMessageBox.critical(
                self,
                self.tr("Component Error"),
                self.tr("Core components for analysis are missing. This tab is disabled."),
            )

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        self.analysis_tabs = QTabWidget()

        self.sa_tab = RedesignedSensitivityWidget(self)
        uq_tab = self._create_uq_tab()

        self.sa_tab_index = self.analysis_tabs.addTab(
            self.sa_tab, QIcon.fromTheme("view-statistics"), ""
        )
        self.uq_tab_index = self.analysis_tabs.addTab(
            uq_tab, QIcon.fromTheme("view-process-users"), ""
        )
        dca_tab = self._create_dca_tab()
        self.dca_tab_index = self.analysis_tabs.addTab(
            dca_tab, QIcon.fromTheme("view-calendar-week"), ""
        )

        main_layout.addWidget(self.analysis_tabs)

    def changeEvent(self, event: QEvent):
        if event.type() == QEvent.Type.LanguageChange:
            self.retranslateUi()
        super().changeEvent(event)

    def retranslateUi(self):
        self.analysis_tabs.setTabText(self.sa_tab_index, self.tr("Sensitivity Analysis"))
        self.analysis_tabs.setTabText(self.uq_tab_index, self.tr("Uncertainty Quantification"))
        self.analysis_tabs.setTabText(self.dca_tab_index, self.tr("Decline Curve Analysis"))

        # UQ Tab
        self.uq_params_group.setTitle(self.tr("1. Uncertain Parameters"))
        self.uq_params_table.setHorizontalHeaderLabels(
            [self.tr("Parameter"), self.tr("Distribution"), self.tr("Details")]
        )
        self.uq_add_btn.setText(self.tr("Add"))
        self.uq_edit_btn.setText(self.tr("Edit"))
        self.uq_remove_btn.setText(self.tr("Remove"))
        self.uq_run_group.setTitle(self.tr("2. Run Analysis"))
        self.uq_analysis_type_label.setText(self.tr("Analysis Type:"))
        self.uq_mc_samples_label.setText(self.tr("MC Samples:"))
        self.uq_pce_order_label.setText(self.tr("PCE Order:"))

        current_uq_type = self.uq_analysis_type_combo.currentData()
        self.uq_analysis_type_combo.clear()
        self.uq_analysis_type_combo.addItem(self.tr("Monte Carlo"), "Monte Carlo")
        self.uq_analysis_type_combo.addItem(self.tr("Polynomial Chaos Expansion (PCE)"), "PCE")
        if current_uq_type:
            self.uq_analysis_type_combo.setCurrentIndex(
                self.uq_analysis_type_combo.findData(current_uq_type)
            )

        self.uq_run_button.setText(self.tr("Run UQ Analysis"))
        self.uq_results_group.setTitle(self.tr("UQ Results"))

        # DCA Tab
        self.dca_well_label.setText(self.tr("Well:"))
        self.dca_model_label.setText(self.tr("Model:"))
        self.dca_forecast_label.setText(self.tr("Forecast Years:"))
        self.dca_run_button.setText(self.tr("Run DCA"))

        if not QWebEngineView:
            plot_disabled_msg = self.tr("Plotting requires PyQt6-WebEngine.")
            self.uq_plot_view.setText(plot_disabled_msg)
            self.dca_plot_view.setText(plot_disabled_msg)

    def update_analyzer_and_engine(
        self,
        analyzer: Optional[SensitivityAnalyzer],
        engine: Optional[OptimizationEngine],
        well_data: Optional[List[Any]] = None,
    ):
        self.analyzer = analyzer
        self.sa_tab.update_analyzer(analyzer)

        self.engine = engine
        self.uq_run_button.setEnabled(self.engine is not None)
        if self.engine:
            self.uq_all_params = self.engine.get_configurable_parameters_for_uq()
            self._uq_rebuild_engine()
            logger.info("AnalysisWidget UQ engine instance updated.")
        else:
            self.uq_all_params.clear()
            self.uq_params_table.setRowCount(0)
            self.uq_engine = None
            logger.warning("AnalysisWidget UQ engine instance removed.")

        self.well_data = well_data
        self.dca_well_combo.clear()
        if self.well_data:
            for well in self.well_data:
                self.dca_well_combo.addItem(well.name, userData=well)

            optimized_well_index = self.dca_well_combo.findText("Field (Optimized)")
            if optimized_well_index != -1:
                self.dca_well_combo.setCurrentIndex(optimized_well_index)

    def _dca_run_analysis(self):
        if not self.well_data:
            QMessageBox.warning(
                self, self.tr("No Data"), self.tr("No well data available for analysis.")
            )
            return

        if self.dca_worker and self.dca_worker.isRunning():
            QMessageBox.warning(
                self, self.tr("Busy"), self.tr("A decline curve analysis is already in progress.")
            )
            return

        well_data = self.dca_well_combo.currentData()
        if not well_data:
            QMessageBox.warning(
                self, self.tr("Input Error"), self.tr("Please select a well to analyze.")
            )
            return

        model_type = self.dca_model_combo.currentText().lower()
        forecast_years = self.dca_forecast_spin.value()

        self.dca_run_button.setEnabled(False)
        self.dca_summary_browser.setText(f"<i>{self.tr('Running analysis...')}</i>")
        if isinstance(self.dca_plot_view, QWebEngineView):
            self.dca_plot_view.setHtml("")

        self.dca_worker = DCAWorker(self.dca_analyzer, well_data, model_type, forecast_years)
        self.dca_worker.signals.result.connect(self._dca_on_result)
        self.dca_worker.signals.error.connect(self._dca_on_error)
        self.dca_worker.signals.finished.connect(self._dca_on_worker_finished)
        self.dca_worker.start()

    def _dca_on_result(self, result: DCAResult):
        self.dca_results = result
        self.dca_summary_browser.clear()
        summary_html = f"<h3>{self.tr('Decline Curve Analysis Results')}</h3>"
        summary_html += f"<p><b>{self.tr('Model Type')}:</b> {result.model_type.capitalize()}</p>"
        summary_html += f"<p><b>{self.tr('R-squared')}:</b> {result.r_squared:.4f}</p>"
        summary_html += f"<p><b>{self.tr('Economic Life')}:</b> {result.economic_life:.2f} {self.tr('years')}</p>"
        summary_html += f"<h4>{self.tr('Parameters')}</h4><ul>"
        for name, value in result.parameters.items():
            summary_html += f"<li><b>{name}:</b> {value:.4f}</li>"
        summary_html += "</ul>"
        self.dca_summary_browser.setHtml(summary_html)

        logger.info(f"DCA plot view is of type: {type(self.dca_plot_view)}")

        if isinstance(self.dca_plot_view, QWebEngineView):
            if self.dca_analyzer:
                fig = self.dca_analyzer.plot_decline_curve(result)
                self.dca_plot_view.setHtml(fig.to_html(include_plotlyjs="cdn"))

    def _dca_on_error(self, error_msg: str):
        self.dca_summary_browser.setHtml(
            f"<p style='color:red;'><b>{self.tr('Error:')}</b><br>{error_msg}</p>"
        )
        QMessageBox.critical(self, self.tr("DCA Error"), error_msg)

    def _dca_on_worker_finished(self):
        self.dca_run_button.setEnabled(True)
        self.dca_worker.deleteLater()
        self.dca_worker = None

    def _uq_rebuild_engine(self):
        if not self.engine:
            return
        config = {
            "parameters": [
                self.uq_params_table.item(r, 0).data(Qt.ItemDataRole.UserRole)
                for r in range(self.uq_params_table.rowCount())
            ]
        }
        self.uq_engine = UncertaintyQuantificationEngine(self.engine, config)
        logger.info("UQEngine instance rebuilt.")

    def _uq_add_parameter(self):
        existing = [
            self.uq_params_table.item(r, 0).data(Qt.ItemDataRole.UserRole)["path"]
            for r in range(self.uq_params_table.rowCount())
        ]
        dialog = EditUQParameterDialog(
            self.uq_all_params,
            existing,
            self.SUPPORTED_DISTRIBUTIONS,
            self.DIST_PARAMS_CONFIG,
            parent=self,
        )
        if dialog.exec() and (data := dialog.get_data()):
            row = self.uq_params_table.rowCount()
            self.uq_params_table.insertRow(row)
            self._uq_update_table_row(row, data)
            self._uq_rebuild_engine()

    def _uq_edit_parameter(self):
        if (row := self.uq_params_table.currentRow()) < 0:
            return
        data = self.uq_params_table.item(row, 0).data(Qt.ItemDataRole.UserRole)
        dialog = EditUQParameterDialog(
            self.uq_all_params,
            [],
            self.SUPPORTED_DISTRIBUTIONS,
            self.DIST_PARAMS_CONFIG,
            data,
            self,
        )
        if dialog.exec() and (new_data := dialog.get_data()):
            self._uq_update_table_row(row, new_data)
            self._uq_rebuild_engine()

    def _uq_remove_parameter(self):
        if (row := self.uq_params_table.currentRow()) >= 0:
            self.uq_params_table.removeRow(row)
            self._uq_rebuild_engine()

    def _uq_update_table_row(self, row: int, data: Dict):
        item = QTableWidgetItem(data["name"])
        item.setData(Qt.ItemDataRole.UserRole, data)
        self.uq_params_table.setItem(row, 0, item)
        self.uq_params_table.setItem(row, 1, QTableWidgetItem(data["distribution"]))
        self.uq_params_table.setItem(
            row, 2, QTableWidgetItem(", ".join(f"{p:.3g}" for p in data["params"]))
        )

    def _uq_run_analysis(self):
        if not self.uq_engine or not self.engine:
            return

        if not self.engine.results or "optimized_params_final_clipped" not in self.engine.results:
            QMessageBox.warning(
                self,
                self.tr("Run Optimization First"),
                self.tr(
                    "An optimization must be run successfully before starting a UQ analysis to establish baseline parameters."
                ),
            )
            return

        fixed_params = self.engine.results["optimized_params_final_clipped"]
        analysis_type = self.uq_analysis_type_combo.currentData()

        if "Monte" in analysis_type:
            method_name = "run_mc_analysis"
            kwargs = {
                "num_samples": self.uq_num_samples_spin.value(),
                "fixed_eor_params": fixed_params,
            }
        else:
            method_name = "run_pce_analysis"
            kwargs = {
                "poly_order": self.uq_pce_order_spin.value(),
                "fixed_eor_params": fixed_params,
            }

        self.uq_run_button.setEnabled(False)
        self.uq_summary_browser.setText(f"<i>{self.tr('Running analysis...')}</i>")
        if isinstance(self.uq_plot_view, QWebEngineView):
            self.uq_plot_view.setHtml("")

        self.uq_worker = UQWorker(self.uq_engine, method_name, kwargs)
        self.uq_worker.result_ready.connect(self._uq_on_result)
        self.uq_worker.error_occurred.connect(self._uq_on_error)
        self.uq_worker.finished.connect(self._uq_on_worker_finished)
        self.uq_worker.start()

    def _uq_on_result(self, result: Any):
        self.uq_summary_browser.clear()
        if isinstance(result, pd.DataFrame):  # MC Results
            html_table = result.describe().transpose().to_html()
            self.uq_summary_browser.setHtml(
                f"<b>{self.tr('Monte Carlo Results:')}</b><br>{html_table}"
            )
            if self.uq_engine and isinstance(self.uq_plot_view, QWebEngineView):
                if fig := self.uq_engine.plot_mc_results(result):
                    self.uq_plot_view.setHtml(fig.to_html(include_plotlyjs="cdn"))
        elif isinstance(result, dict):  # PCE Results
            items_html = "".join(
                f"<li><b>{k.replace('_', ' ').title()}:</b> {v}</li>" for k, v in result.items()
            )
            html = f"<b>{self.tr('PCE Results:')}</b><ul>{items_html}</ul>"
            self.uq_summary_browser.setHtml(html)
            if self.uq_engine and isinstance(self.uq_plot_view, QWebEngineView):
                if fig := self.uq_engine.plot_pce_sobol_indices(result):
                    self.uq_plot_view.setHtml(fig.to_html(include_plotlyjs="cdn"))

    def _uq_on_error(self, error_msg: str):
        self.uq_summary_browser.setHtml(
            f"<p style='color:red;'><b>{self.tr('Error:')}</b><br>{error_msg}</p>"
        )
        QMessageBox.critical(self, self.tr("UQ Error"), error_msg)

    def _uq_on_worker_finished(self):
        self.uq_run_button.setEnabled(self.engine is not None)
        self.uq_worker.deleteLater()
        self.uq_worker = None

    def get_uq_parameters(self) -> List[Dict]:
        """Returns the current list of UQ parameter configurations."""
        return [
            self.uq_params_table.item(r, 0).data(Qt.ItemDataRole.UserRole)
            for r in range(self.uq_params_table.rowCount())
        ]

    def set_uq_parameters(self, uq_params: List[Dict]):
        """Populates the UQ parameters table from loaded data."""
        self.uq_params_table.setRowCount(0)
        if uq_params:
            for param_data in uq_params:
                row = self.uq_params_table.rowCount()
                self.uq_params_table.insertRow(row)
                self._uq_update_table_row(row, param_data)
        self._uq_rebuild_engine()

    # --- Uncertainty Quantification Section ---

    def _create_uq_tab(self) -> QWidget:
        uq_widget = QWidget()
        main_layout = QVBoxLayout(uq_widget)
        splitter = QSplitter(Qt.Orientation.Horizontal)

        config_container = QWidget()
        config_layout = QVBoxLayout(config_container)
        self.uq_params_group = QGroupBox()
        params_layout = QVBoxLayout(self.uq_params_group)
        self.uq_params_table = QTableWidget(0, 3)
        self.uq_params_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.uq_params_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.uq_params_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Stretch
        )
        params_layout.addWidget(self.uq_params_table)
        btn_layout = QHBoxLayout()
        self.uq_add_btn = QPushButton("Add")
        self.uq_edit_btn = QPushButton("Edit")
        self.uq_remove_btn = QPushButton("Remove")
        btn_layout.addWidget(self.uq_add_btn)
        btn_layout.addWidget(self.uq_edit_btn)
        btn_layout.addWidget(self.uq_remove_btn)
        btn_layout.addStretch()
        params_layout.addLayout(btn_layout)
        config_layout.addWidget(self.uq_params_group)

        self.uq_run_group = QGroupBox()
        run_grid = QGridLayout(self.uq_run_group)
        self.uq_analysis_type_label = QLabel()
        self.uq_analysis_type_combo = QComboBox()
        run_grid.addWidget(self.uq_analysis_type_label, 0, 0)
        run_grid.addWidget(self.uq_analysis_type_combo, 0, 1)

        self.uq_mc_samples_label = QLabel()
        self.uq_num_samples_spin = QSpinBox()
        self.uq_num_samples_spin.setRange(10, 100_000)
        self.uq_num_samples_spin.setValue(1000)
        run_grid.addWidget(self.uq_mc_samples_label, 1, 0)
        run_grid.addWidget(self.uq_num_samples_spin, 1, 1)

        self.uq_pce_order_label = QLabel()
        self.uq_pce_order_spin = QSpinBox()
        self.uq_pce_order_spin.setRange(1, 5)
        self.uq_pce_order_spin.setValue(2)
        run_grid.addWidget(self.uq_pce_order_label, 2, 0)
        run_grid.addWidget(self.uq_pce_order_spin, 2, 1)

        self.uq_run_button = QPushButton("Run UQ Analysis")
        self.uq_run_button.setEnabled(False)
        run_grid.addWidget(self.uq_run_button, 3, 0, 1, 2)
        config_layout.addWidget(self.uq_run_group)
        config_layout.addStretch()

        results_container = QWidget()
        results_layout = QVBoxLayout(results_container)
        self.uq_results_group = QGroupBox()
        results_box = QVBoxLayout(self.uq_results_group)
        self.uq_plot_view = QWebEngineView() if QWebEngineView else QLabel()
        self.uq_summary_browser = QTextBrowser()
        results_box.addWidget(self.uq_plot_view, 1)
        results_box.addWidget(self.uq_summary_browser)
        results_layout.addWidget(self.uq_results_group)

        splitter.addWidget(config_container)
        config_container.setMinimumWidth(300)
        config_container.setMaximumWidth(450)
        splitter.addWidget(results_container)
        results_container.setMinimumWidth(400)
        # Use stretch factors for better resizing behavior
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([350, 750])  # Initial sizes only
        main_layout.addWidget(splitter)
        return uq_widget

    def _create_dca_tab(self) -> QWidget:
        dca_widget = QWidget()
        main_layout = QVBoxLayout(dca_widget)
        splitter = QSplitter(Qt.Orientation.Horizontal)

        config_container = QWidget()
        config_layout = QFormLayout(config_container)

        self.dca_well_label = QLabel("Well:")
        self.dca_well_combo = QComboBox()
        config_layout.addRow(self.dca_well_label, self.dca_well_combo)

        self.dca_model_label = QLabel("Model:")
        self.dca_model_combo = QComboBox()
        self.dca_model_combo.addItems(["Auto", "Exponential", "Hyperbolic", "Harmonic"])
        config_layout.addRow(self.dca_model_label, self.dca_model_combo)

        self.dca_forecast_label = QLabel("Forecast Years:")
        self.dca_forecast_spin = QSpinBox()
        self.dca_forecast_spin.setRange(1, 100)
        self.dca_forecast_spin.setValue(20)
        config_layout.addRow(self.dca_forecast_label, self.dca_forecast_spin)

        self.dca_run_button = QPushButton("Run DCA")
        config_layout.addRow(self.dca_run_button)

        results_container = QWidget()
        results_layout = QVBoxLayout(results_container)
        results_splitter = QSplitter(Qt.Orientation.Vertical)

        self.dca_plot_view = QWebEngineView() if QWebEngineView else QLabel()
        if not QWebEngineView:
            self.dca_plot_view.setText("Plotting requires PyQt6-WebEngine.")
            self.dca_plot_view.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.dca_summary_browser = QTextBrowser()

        results_splitter.addWidget(self.dca_plot_view)
        results_splitter.addWidget(self.dca_summary_browser)
        # Use stretch factors for better resizing behavior
        results_splitter.setStretchFactor(0, 1)
        results_splitter.setStretchFactor(1, 0)
        results_splitter.setSizes([400, 150])  # Initial sizes only

        results_layout.addWidget(results_splitter)

        splitter.addWidget(config_container)
        config_container.setMinimumWidth(250)
        config_container.setMaximumWidth(350)
        splitter.addWidget(results_container)
        results_container.setMinimumWidth(400)
        # Use stretch factors for better resizing behavior
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([300, 700])  # Initial sizes only
        main_layout.addWidget(splitter)

        return dca_widget

    def _on_dca_well_changed(self, index: int):
        """Checks if the currently selected well in the DCA tab has valid data and updates the UI."""
        if index == -1:
            self.dca_run_button.setEnabled(False)
            self.dca_run_button.setToolTip(self.tr("No well selected."))
            return

        well_data = self.dca_well_combo.itemData(index)

        # Check if well_data is an object with a 'properties' attribute which is a dictionary
        if hasattr(well_data, "properties") and isinstance(getattr(well_data, "properties"), dict):
            properties = well_data.properties
            if "time" in properties and "rate" in properties:
                self.dca_run_button.setEnabled(True)
                self.dca_run_button.setToolTip(
                    self.tr("Run Decline Curve Analysis on the selected well/profile.")
                )
            else:
                self.dca_run_button.setEnabled(False)
                self.dca_run_button.setToolTip(
                    self.tr(
                        "The selected well does not contain production data ('time' and 'rate' profiles)."
                    )
                )
        else:
            self.dca_run_button.setEnabled(False)
            self.dca_run_button.setToolTip(self.tr("Select a well with valid production data."))

    # --- Signal Connections ---

    def _connect_signals(self):
        # SA Signals
        self.sa_tab.help_requested.connect(self.help_requested)

        # UQ Signals
        self.uq_add_btn.clicked.connect(self._uq_add_parameter)
        self.uq_edit_btn.clicked.connect(self._uq_edit_parameter)
        self.uq_params_table.doubleClicked.connect(self._uq_edit_parameter)
        self.uq_remove_btn.clicked.connect(self._uq_remove_parameter)
        self.uq_run_button.clicked.connect(self._uq_run_analysis)

        # DCA Signals
        self.dca_run_button.clicked.connect(self._dca_run_analysis)
        self.dca_well_combo.currentIndexChanged.connect(self._on_dca_well_changed)

    def update_graphs(self):
        """Public method to be called after loading a project to refresh all graphs."""
        logger.info("AnalysisWidget explicitly requested to update all graphs.")
        if self.analyzer and self.analyzer.sensitivity_run_data:
            df = pd.DataFrame(self.analyzer.sensitivity_run_data)
        if self.uq_engine and self.uq_engine.results:
            self._uq_on_result(self.uq_engine.results)
