import logging
from typing import Optional, Any, Dict
from dataclasses import fields

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QGroupBox, QLabel,
    QComboBox, QPushButton, QTableWidget, QTableWidgetItem, QHeaderView,
    QSplitter, QAbstractItemView, QMessageBox, QSpinBox, QDoubleSpinBox, QFormLayout
)
from PyQt6.QtGui import QIcon
from PyQt6.QtCore import Qt, pyqtSignal
import pandas as pd
from PyQt6.QtWebEngineWidgets import QWebEngineView

from .workers.optimization_worker import OptimizationWorker
from co2eor_optimizer.core.optimisation_engine import OptimizationEngine
from co2eor_optimizer.core.data_models import GeneticAlgorithmParams

logger = logging.getLogger(__name__)

OPTIMIZATION_METHODS = {
    "Hybrid (GA -> Bayesian)": "hybrid_optimize",
    "Genetic Algorithm": "optimize_genetic_algorithm",
    "Bayesian Optimization": "optimize_bayesian",
}
OPTIMIZATION_OBJECTIVES = {
    "Net Present Value (NPV)": "npv",
    "Recovery Factor (RF)": "recovery_factor",
    "CO2 Utilization": "co2_utilization",
}

class OptimizationWidget(QWidget):
    """Tab for configuring, running, and viewing EOR optimization tasks."""
    optimization_completed = pyqtSignal(dict)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.engine: Optional[OptimizationEngine] = None
        self.worker: Optional[OptimizationWorker] = None
        self.ga_param_inputs: Dict[str, QWidget] = {}

        self._setup_ui()
        self._connect_signals()

        if not all([OptimizationWorker, OptimizationEngine, GeneticAlgorithmParams]):
            self.setEnabled(False)
            QMessageBox.critical(self, "Component Error", "Core components for optimization are missing. This tab will be disabled.")

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        config_panel = self._create_config_panel()
        results_panel = self._create_results_panel()

        splitter.addWidget(config_panel)
        splitter.addWidget(results_panel)
        splitter.setSizes([400, 600])
        main_layout.addWidget(splitter)

    def _create_config_panel(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)

        setup_group = QGroupBox("Optimization Setup")
        setup_grid = QGridLayout(setup_group)
        self.method_combo = QComboBox()
        self.method_combo.addItems(OPTIMIZATION_METHODS.keys())
        self.objective_combo = QComboBox()
        self.objective_combo.addItems(OPTIMIZATION_OBJECTIVES.keys())
        setup_grid.addWidget(QLabel("Method:"), 0, 0)
        setup_grid.addWidget(self.method_combo, 0, 1)
        setup_grid.addWidget(QLabel("Objective:"), 1, 0)
        setup_grid.addWidget(self.objective_combo, 1, 1)
        layout.addWidget(setup_group)

        self.ga_params_group = QGroupBox("Genetic Algorithm Parameters")
        self.ga_params_form = QFormLayout(self.ga_params_group)
        self.ga_params_group.setVisible(False) # Hidden by default
        layout.addWidget(self.ga_params_group)
        
        layout.addStretch()
        self.run_button = QPushButton(QIcon.fromTheme("system-run"), "Run Optimization")
        self.run_button.setEnabled(False)
        layout.addWidget(self.run_button, 0, Qt.AlignmentFlag.AlignRight)
        return container

    def _create_results_panel(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        results_group = QGroupBox("Optimization Results")
        results_layout = QVBoxLayout(results_group)
        
        self.summary_label = QLabel("<i>Run optimization to view results.</i>")
        self.summary_label.setWordWrap(True)
        results_layout.addWidget(self.summary_label)

        self.results_table = QTableWidget()
        self.results_table.setColumnCount(2)
        self.results_table.setHorizontalHeaderLabels(["Parameter", "Optimized Value"])
        self.results_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.results_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.results_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        results_layout.addWidget(self.results_table)

        self.plot_view = QWebEngineView()
        results_layout.addWidget(self.plot_view, 1)
        
        layout.addWidget(results_group)
        return container

    def _connect_signals(self):
        self.method_combo.currentTextChanged.connect(self._on_method_changed)
        self.run_button.clicked.connect(self._run_optimization)

    def update_engine(self, engine: Optional[OptimizationEngine]):
        """Public method for MainWindow to provide the optimization engine."""
        self.engine = engine
        self.run_button.setEnabled(self.engine is not None)
        if self.engine:
            self._on_method_changed(self.method_combo.currentText())
            logger.info("OptimizationWidget engine instance updated.")
        else:
            self.ga_params_group.setVisible(False)
            logger.warning("OptimizationWidget engine instance removed.")

    def _on_method_changed(self, method_name: str):
        show_ga = "Genetic Algorithm" in method_name or "Hybrid" in method_name
        self.ga_params_group.setVisible(show_ga)
        if show_ga and self.engine:
            self._populate_ga_form(self.engine.ga_params_default_config)

    def _populate_ga_form(self, ga_params: GeneticAlgorithmParams):
        # Clear existing form
        while self.ga_params_form.rowCount() > 0:
            self.ga_params_form.removeRow(0)
        self.ga_param_inputs.clear()

        for field in fields(ga_params):
            label = field.name.replace("_", " ").title()
            value = getattr(ga_params, field.name)
            if field.type is int:
                widget = QSpinBox()
                widget.setRange(0, 1_000_000)
                widget.setValue(value)
            elif field.type is float:
                widget = QDoubleSpinBox()
                widget.setRange(0.0, 1_000_000.0)
                widget.setDecimals(4)
                widget.setValue(value)
            else: continue
            
            self.ga_param_inputs[field.name] = widget
            self.ga_params_form.addRow(label, widget)

    def _get_ga_params_from_form(self) -> Optional[GeneticAlgorithmParams]:
        if not GeneticAlgorithmParams: return None
        try:
            kwargs = {name: w.value() for name, w in self.ga_param_inputs.items()}
            return GeneticAlgorithmParams(**kwargs)
        except Exception as e:
            QMessageBox.critical(self, "GA Settings Error", f"Invalid GA parameter input: {e}")
            logger.error(f"Error creating GeneticAlgorithmParams from form: {e}", exc_info=True)
            return None

    def _run_optimization(self):
        if not self.engine:
            QMessageBox.critical(self, "Engine Error", "Optimization Engine is not available.")
            return
        if self.worker and self.worker.isRunning():
            QMessageBox.warning(self, "Busy", "An optimization run is already in progress.")
            return

        method_key = self.method_combo.currentText()
        method_name = OPTIMIZATION_METHODS[method_key]
        self.engine.chosen_objective = OPTIMIZATION_OBJECTIVES[self.objective_combo.currentText()]

        kwargs = {}
        if self.ga_params_group.isVisible():
            ga_params = self._get_ga_params_from_form()
            if not ga_params: return
            kwargs["ga_params_override"] = ga_params

        self._clear_results()
        self.summary_label.setText("<i>Running optimization, please wait...</i>")
        self.run_button.setEnabled(False)

        self.worker = OptimizationWorker(self.engine, method_name, kwargs)
        self.worker.result_ready.connect(self._on_result)
        self.worker.error_occurred.connect(self._on_error)
        self.worker.finished.connect(self._on_worker_finished)
        self.worker.progress_updated.connect(lambda p, m: self.summary_label.setText(f"<i>Status: {m} ({p}%)</i>"))
        self.worker.start()

    def _clear_results(self):
        self.summary_label.setText("<i>Run optimization to view results.</i>")
        self.results_table.setRowCount(0)
        self.plot_view.setHtml("")

    def _on_result(self, results: Dict[str, Any]):
        self._display_summary(results)
        self._display_table(results.get('optimized_params_final_clipped', {}))
        self._display_plot(results)
        self.optimization_completed.emit(results)

    def _display_summary(self, results: Dict[str, Any]):
        obj_val = results.get('objective_function_value', 'N/A')
        obj_name = results.get('chosen_objective', 'N/A').replace("_", " ").title()
        rf = results.get('final_recovery_factor_reported', 'N/A')
        method = results.get('method', 'N/A').replace("_", " ").title()
        
        summary = (
            f"<b>Optimization Complete: {method}</b><br>"
            f"Final Objective ({obj_name}): <b>{obj_val:.4g if isinstance(obj_val, float) else obj_val}</b><br>"
            f"Final Recovery Factor: <b>{rf:.4f if isinstance(rf, float) else rf}</b>"
        )
        self.summary_label.setText(summary)

    def _display_table(self, params: Dict[str, Any]):
        self.results_table.setRowCount(len(params))
        for i, (key, val) in enumerate(params.items()):
            self.results_table.setItem(i, 0, QTableWidgetItem(key.replace("_", " ").title()))
            self.results_table.setItem(i, 1, QTableWidgetItem(f"{val:.4g}" if isinstance(val, float) else str(val)))
        self.results_table.resizeRowsToContents()
        
    def _display_plot(self, results: Dict[str, Any]):
        if self.engine and hasattr(self.engine, 'plot_optimization_convergence'):
            try:
                fig = self.engine.plot_optimization_convergence(results)
                self.plot_view.setHtml(fig.to_html(include_plotlyjs='cdn'))
            except Exception as e:
                logger.error(f"Error generating convergence plot: {e}", exc_info=True)
                self.plot_view.setHtml(f"<p style='color:red;'>Plotting Error: {e}</p>")

    def _on_error(self, error_msg: str):
        self._clear_results()
        self.summary_label.setHtml(f"<p style='color:red;'><b>Error:</b><br>{error_msg}</p>")

    def _on_worker_finished(self):
        self.run_button.setEnabled(self.engine is not None)
        if self.worker:
            self.worker.deleteLater()
            self.worker = None