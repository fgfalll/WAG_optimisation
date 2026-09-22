from __future__ import annotations # This line is crucial for forward references and conditional imports
import logging
from typing import Optional, Any, Dict, List, Union, Tuple
import numpy as np
import pandas as pd

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QGridLayout, QGroupBox, QLabel, QLineEdit,
    QComboBox, QPushButton, QSplitter, QTabWidget, QTreeView,
    QAbstractItemView, QTableWidget, QTableWidgetItem, QHeaderView,
    QMessageBox, QSpinBox
)
from PyQt6.QtGui import QIcon, QStandardItemModel, QStandardItem
from PyQt6.QtCore import pyqtSignal, Qt

# Initialize these to None. They will be assigned the actual classes
# or remain None if imports fail. This helps type checkers understand the potential
# absence of these types, especially when using 'Optional' with them.
SensitivityAnalysisWorker: Optional[Any] = None
SensitivityAnalyzer: Optional[Any] = None
OptimizationEngine: Optional[Any] = None
# Initialize QWebEngineView here as well, and use Any for its type hint
QWebEngineView: Optional[Any] = None

try:
    from .workers.sensitivity_analysis_worker import SensitivityAnalysisWorker
    from co2eor_optimizer.analysis.sensitivity_analyzer import SensitivityAnalyzer
    from co2eor_optimizer.core.optimisation_engine import OptimizationEngine
except ImportError as e:
    logging.critical(f"SensitivityWidget: Failed to import core optimizer components: {e}.")

try:
    # Attempt to import QWebEngineView. If it fails, it remains None.
    from PyQt6.QtWebEngineWidgets import QWebEngineView as ImportedQWebEngineView
    QWebEngineView = ImportedQWebEngineView # Assign to our module-level variable
except ImportError as e:
    logging.critical(f"SensitivityWidget: Failed to import PyQt6-WebEngineWidgets: {e}. Plotting features may be disabled.")
    # QWebEngineView remains None as initialized at the top

logger = logging.getLogger(__name__)

class ParameterSelectionTreeView(QTreeView):
    """A specialized QTreeView for selecting parameters with check boxes."""
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.model = QStandardItemModel(self)
        self.setModel(self.model)
        self.setHeaderHidden(True)
        self.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)

    def populate(self, param_structure: Dict[str, List[Tuple[str, str]]]):
        """Populates the tree with parameters from a structured dictionary."""
        self.model.clear()
        root = self.model.invisibleRootItem()
        for category, params in param_structure.items():
            cat_item = QStandardItem(category)
            cat_item.setEditable(False)
            cat_item.setSelectable(False)
            for path, name in params: # Correctly unpacks (internal_name, display_name)
                param_item = QStandardItem(name)
                param_item.setData(path, Qt.ItemDataRole.UserRole)
                param_item.setCheckable(True)
                param_item.setEditable(False)
                cat_item.appendRow(param_item)
            root.appendRow(cat_item)
        self.expandAll()

    def get_selected_paths(self) -> List[str]:
        """Returns the full paths of all checked parameters."""
        paths = []
        root = self.model.invisibleRootItem()
        for i in range(root.rowCount()):
            cat_item = root.child(i)
            if cat_item is None: continue
            for j in range(cat_item.rowCount()):
                param_item = cat_item.child(j)
                if param_item is None: continue
                if param_item.checkState() == Qt.CheckState.Checked:
                    paths.append(param_item.data(Qt.ItemDataRole.UserRole))
        return paths

class SensitivityWidget(QWidget):
    """Tab for configuring and running Sensitivity Analysis (SA)."""
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.analyzer: Optional['SensitivityAnalyzer'] = None
        self.worker: Optional['SensitivityAnalysisWorker'] = None
        
        # Explicitly declare attributes for type checkers and clarity
        self.tornado_view: QWidget
        self.spider_view: QWidget
        self.surface_3d_view: QWidget
        self.data_table: QTableWidget
        self.results_tabs: QTabWidget
        self.run_button: QPushButton
        self.param_tree: ParameterSelectionTreeView
        self.sa_type_combo: QComboBox
        self.num_steps_spin: QSpinBox
        self.variation_edit: QLineEdit
        self.objective_combo: QComboBox

        self._setup_ui()
        self._connect_signals()

        if SensitivityAnalysisWorker is None or SensitivityAnalyzer is None:
            self.setEnabled(False)
            QMessageBox.critical(self, "Component Error", "Core components for SA are missing. This tab is disabled.")
        else:
            self.run_button.setEnabled(self.analyzer is not None)

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        splitter = QSplitter(Qt.Orientation.Horizontal)

        config_panel = self._create_config_panel()
        results_panel = self._create_results_panel()

        splitter.addWidget(config_panel)
        splitter.addWidget(results_panel)
        splitter.setSizes([450, 650])
        main_layout.addWidget(splitter)
        
        # Call this after all UI elements have been created to avoid AttributeErrors
        self._on_type_changed(self.sa_type_combo.currentText())

    def _create_config_panel(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        
        param_group = QGroupBox("1. Select Parameters")
        param_layout = QVBoxLayout(param_group)
        self.param_tree = ParameterSelectionTreeView()
        param_layout.addWidget(self.param_tree)
        layout.addWidget(param_group, 1)

        config_group = QGroupBox("2. Configure Analysis")
        config_grid = QGridLayout(config_group)

        config_grid.addWidget(QLabel("Analysis Type:"), 0, 0)
        self.sa_type_combo = QComboBox()
        self.sa_type_combo.addItems(["One-Way", "Two-Way", "Re-Optimization"])
        config_grid.addWidget(self.sa_type_combo, 0, 1)

        config_grid.addWidget(QLabel("Number of Steps:"), 1, 0)
        self.num_steps_spin = QSpinBox()
        self.num_steps_spin.setRange(3, 101)
        self.num_steps_spin.setValue(11)
        config_grid.addWidget(self.num_steps_spin, 1, 1)

        config_grid.addWidget(QLabel("Variation:"), 2, 0)
        self.variation_edit = QLineEdit()
        config_grid.addWidget(self.variation_edit, 2, 1)

        config_grid.addWidget(QLabel("Objective:"), 3, 0)
        self.objective_combo = QComboBox()
        self.objective_combo.addItems(['npv', 'recovery_factor', 'co2_utilization'])
        self.objective_combo.setToolTip("Select the single objective to analyze and plot.")
        config_grid.addWidget(self.objective_combo, 3, 1)

        self.run_button = QPushButton(QIcon.fromTheme("system-run"), "Run Sensitivity Analysis")
        self.run_button.setEnabled(False)
        config_grid.addWidget(self.run_button, 4, 0, 1, 2)
        layout.addWidget(config_group)
        
        return container

    def _create_results_panel(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        self.results_tabs = QTabWidget()
        
        if QWebEngineView:
            self.tornado_view = QWebEngineView()
            self.spider_view = QWebEngineView()
            self.surface_3d_view = QWebEngineView()
        else:
            self.tornado_view = QLabel("Plotting functionality (Tornado Plot) requires PyQt6-WebEngine. Please install it.")
            self.spider_view = QLabel("Plotting functionality (Spider Plot) requires PyQt6-WebEngine. Please install it.")
            self.surface_3d_view = QLabel("Plotting functionality (3D Plot) requires PyQt6-WebEngine. Please install it.")
            for label in [self.tornado_view, self.spider_view, self.surface_3d_view]:
                label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        self.data_table = QTableWidget()
        self.data_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.data_table.setAlternatingRowColors(True)

        self.results_tabs.addTab(self.tornado_view, "Tornado Plot")
        self.results_tabs.addTab(self.spider_view, "Spider Plot")
        self.results_tabs.addTab(self.surface_3d_view, "3D Surface Plot")
        self.results_tabs.addTab(self.data_table, "Data Table")
        
        layout.addWidget(self.results_tabs)
        return container
    
    def _connect_signals(self):
        self.run_button.clicked.connect(self._run_analysis)
        self.sa_type_combo.currentTextChanged.connect(self._on_type_changed)

    def update_analyzer(self, analyzer: Optional[SensitivityAnalyzer]):
        self.analyzer = analyzer
        self.run_button.setEnabled(self.analyzer is not None)
        if self.analyzer:
            try:
                param_structure = self.analyzer.get_configurable_parameters()
                self.param_tree.populate(param_structure)
                logger.info("SensitivityWidget analyzer instance updated and parameters populated.")
            except Exception as e:
                logger.error(f"Failed to get configurable parameters from analyzer: {e}")
                QMessageBox.warning(self, "Analyzer Error", f"Could not load parameters from analyzer: {e}")
                self.param_tree.model.clear()
        else:
            self.param_tree.model.clear()
            logger.warning("SensitivityWidget analyzer instance removed.")

    def _on_type_changed(self, sa_type: str):
        is_2way_visible = sa_type == "Two-Way"
        plot_tab_index = self.results_tabs.indexOf(self.surface_3d_view)
        if plot_tab_index != -1:
            self.results_tabs.setTabVisible(plot_tab_index, is_2way_visible)

        if not isinstance(self.surface_3d_view, QWebEngineView) and is_2way_visible:
            logger.warning("Attempted to show 3D Surface Plot tab, but PyQt6-WebEngine is not installed.")

        tooltips = {
            "One-Way": "Enter a min/max range (e.g., '1000,2000'), a list of values (e.g., '1000,1500,2000'), or a single value (e.g., '1500') to test a default +/- 20% range.",
            "Two-Way": "Requires exactly two parameters. Format: 'vals_for_param1;vals_for_param2'. Example: '10,20;0.5,0.7'.",
            "Re-Optimization": "Requires exactly one parameter. Provide a comma-separated list of values to test. Example: '1000,1500,2000'."
        }
        self.variation_edit.setToolTip(tooltips.get(sa_type, "Enter variation values based on analysis type."))

        placeholders = {
            "One-Way": "e.g., 1000,2000",
            "Two-Way": "e.g., 10,20;0.5,0.7",
            "Re-Optimization": "e.g., 1000,1500,2000"
        }
        self.variation_edit.setPlaceholderText(placeholders.get(sa_type, ""))

    def _run_analysis(self):
        if not self.analyzer:
            QMessageBox.critical(self, "Analyzer Error", "Sensitivity Analyzer is not available. Please load a project first.")
            return
        if self.worker and self.worker.isRunning():
            QMessageBox.warning(self, "Busy", "An analysis is already in progress. Please wait for it to complete.")
            return

        selected_paths = self.param_tree.get_selected_paths()
        sa_type = self.sa_type_combo.currentText()
        objective = self.objective_combo.currentText()
        
        if not objective:
            QMessageBox.warning(self, "Input Error", "Please select an objective for the analysis.")
            return
        if not selected_paths:
            QMessageBox.warning(self, "Input Error", "Please select at least one parameter for analysis.")
            return

        try:
            objectives = [objective]
            method_name, kwargs = self._prepare_run_args(sa_type, selected_paths, objectives)
        except ValueError as e:
            QMessageBox.warning(self, "Configuration Error", str(e))
            return

        self._clear_results()
        self.results_tabs.setCurrentWidget(self.data_table)
        self.data_table.setColumnCount(1)
        self.data_table.setRowCount(1)
        self.data_table.setItem(0, 0, QTableWidgetItem("<i>Running analysis, please wait...</i>"))
        self.run_button.setEnabled(False)

        if SensitivityAnalysisWorker:
            self.worker = SensitivityAnalysisWorker(self.analyzer, method_name, kwargs)
            self.worker.result_ready.connect(self._on_result)
            self.worker.error_occurred.connect(self._on_error)
            self.worker.finished.connect(self._on_worker_finished)
            self.worker.start()
            logger.info(f"Sensitivity analysis started: Type='{sa_type}', Method='{method_name}'")
        else:
            self._on_error("Sensitivity analysis worker is not available due to missing components.")

    def _prepare_run_args(self, sa_type: str, paths: List[str], objectives: List[str]) -> Tuple[str, Dict]:
        variation_text = self.variation_edit.text()
        if not variation_text:
            raise ValueError("Variation input cannot be empty.")
        
        num_steps = self.num_steps_spin.value()
        
        if sa_type == "One-Way":
            return "run_one_way_sensitivity", {
                "param_paths": paths, "variation_str": variation_text, 
                "num_steps": num_steps, "objectives": objectives
            }
        elif sa_type == "Two-Way":
            if len(paths) != 2:
                raise ValueError("Select exactly two parameters for Two-Way analysis.")
            variations = variation_text.split(';')
            if len(variations) != 2 or not all(v.strip() for v in variations):
                raise ValueError("Two-Way variation requires two value lists separated by ';'.")
            return "run_two_way_sensitivity", {
                "param1_path": paths[0], "param1_values_str": variations[0].strip(),
                "param2_path": paths[1], "param2_values_str": variations[1].strip(),
                "objectives": objectives
            }
        elif sa_type == "Re-Optimization":
            if len(paths) != 1:
                raise ValueError("Select exactly one parameter for Re-Optimization analysis.")
            return "run_reoptimization_sensitivity", {
                "primary_param_to_vary": paths[0], "variation_values_str": variation_text,
                "objectives_at_optimum": objectives
            }
        else:
            raise ValueError(f"Analysis type '{sa_type}' is not yet implemented or recognized.")

    def _clear_results(self):
        if isinstance(self.tornado_view, QWebEngineView): self.tornado_view.setHtml("")
        if isinstance(self.spider_view, QWebEngineView): self.spider_view.setHtml("")
        if isinstance(self.surface_3d_view, QWebEngineView): self.surface_3d_view.setHtml("")
        self.data_table.clear()
        self.data_table.setRowCount(0)
        self.data_table.setColumnCount(0)
        logger.debug("Cleared previous analysis results.")

    def _on_result(self, df: pd.DataFrame):
        self._clear_results()
        if df.empty:
            self.data_table.setRowCount(1)
            self.data_table.setColumnCount(1)
            self.data_table.setItem(0, 0, QTableWidgetItem("Analysis produced no results or an empty DataFrame."))
            logger.warning("Received empty DataFrame from sensitivity analysis worker.")
            return

        self._populate_data_table(df)
        self._generate_plots(df)
        logger.info("Sensitivity analysis results processed and displayed.")

    def _populate_data_table(self, df: pd.DataFrame):
        self.data_table.setRowCount(df.shape[0])
        self.data_table.setColumnCount(df.shape[1])
        self.data_table.setHorizontalHeaderLabels(df.columns)
        for r, row in enumerate(df.itertuples(index=False)):
            for c, val in enumerate(row):
                display_val = f"{val:.4g}" if isinstance(val, (float, np.floating)) else str(val)
                self.data_table.setItem(r, c, QTableWidgetItem(display_val))
        self.data_table.resizeColumnsToContents()
        self.data_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        logger.debug("Data table populated.")

    def _generate_plots(self, df: pd.DataFrame):
        if not QWebEngineView or not isinstance(self.tornado_view, QWebEngineView):
            logger.warning("QWebEngineView not available or plot views are placeholders. Skipping plot generation.")
            if not isinstance(self.tornado_view, QWebEngineView):
                self.tornado_view.setText("<p style='color:red; text-align:center;'>Plotting functionality requires PyQt6-WebEngine, which is not installed.</p>")
            # Do the same for other plot views...
            return

        sa_type = self.sa_type_combo.currentText()
        primary_obj = self.objective_combo.currentText()

        if not primary_obj or primary_obj not in df.columns:
            logger.warning(f"Primary objective '{primary_obj}' not found in results. Cannot generate plots.")
            plot_error_html = f"<p style='color:red; text-align:center;'>Cannot generate plots: Objective '{primary_obj}' not found in results data.</p>"
            self.tornado_view.setHtml(plot_error_html)
            self.spider_view.setHtml(plot_error_html)
            self.surface_3d_view.setHtml(plot_error_html)
            return

        if sa_type == "One-Way":
            try:
                if self.analyzer:
                    tornado_fig = self.analyzer.plot_tornado_chart(df, primary_obj)
                    spider_fig = self.analyzer.plot_spider_chart(df, primary_obj)
                    if tornado_fig: self.tornado_view.setHtml(tornado_fig.to_html(include_plotlyjs='cdn'))
                    if spider_fig: self.spider_view.setHtml(spider_fig.to_html(include_plotlyjs='cdn'))
                    self.results_tabs.setCurrentWidget(self.tornado_view)
                    logger.info("One-Way sensitivity plots generated.")
                else:
                    raise RuntimeError("Analyzer is not available to generate plots.")
            except Exception as e:
                logger.error(f"Error generating one-way plots: {e}", exc_info=True)
                self.tornado_view.setHtml(f"<p style='color:red;'>Plotting Error: {e}</p>")

        elif sa_type == "Two-Way":
            selected_paths = self.param_tree.get_selected_paths()
            if len(selected_paths) != 2: return
            param1_path, param2_path = selected_paths
            try:
                if self.analyzer:
                    surface_fig = self.analyzer.plot_3d_surface(df, param1_path, param2_path, primary_obj)
                    if surface_fig: self.surface_3d_view.setHtml(surface_fig.to_html(include_plotlyjs='cdn'))
                    self.results_tabs.setCurrentWidget(self.surface_3d_view)
                    logger.info("Two-Way sensitivity plot generated.")
                else:
                    raise RuntimeError("Analyzer is not available to generate plots.")
            except Exception as e:
                logger.error(f"Error generating 3D surface plot: {e}", exc_info=True)
                self.surface_3d_view.setHtml(f"<p style='color:red;'>Plotting Error: {e}</p>")

        elif sa_type == "Re-Optimization":
            logger.info("Re-Optimization results received. No specific plots are generated for this type yet.")
            info_html = "<p style='color:blue; text-align:center;'>Re-Optimization results are shown in the Data Table.<br>Specific plotting for this analysis type is under development.</p>"
            if isinstance(self.tornado_view, QWebEngineView): self.tornado_view.setHtml(info_html)
            if isinstance(self.spider_view, QWebEngineView): self.spider_view.setHtml(info_html)
            if isinstance(self.surface_3d_view, QWebEngineView): self.surface_3d_view.setHtml(info_html)

    def _on_error(self, error_msg: str):
        self._clear_results()
        self.data_table.setColumnCount(1)
        self.data_table.setRowCount(1)
        self.data_table.setHorizontalHeaderLabels(["Error"])
        self.data_table.setItem(0, 0, QTableWidgetItem(f"Error: {error_msg}"))
        self.data_table.resizeColumnsToContents()
        self.data_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        QMessageBox.critical(self, "Analysis Error", f"An error occurred during analysis: {error_msg}")
        logger.error(f"Sensitivity analysis worker reported an error: {error_msg}")

    def _on_worker_finished(self):
        self.run_button.setEnabled(self.analyzer is not None)
        if self.worker:
            self.worker.deleteLater()
            self.worker = None
        logger.info("Sensitivity analysis worker finished.")