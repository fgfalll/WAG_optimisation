from __future__ import annotations
import logging
from typing import Optional, Any, Dict, List, Tuple

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QGridLayout, QGroupBox, QLabel, QLineEdit,
    QComboBox, QPushButton, QSplitter, QTabWidget, QTreeView,
    QAbstractItemView, QTableWidget, QTableWidgetItem, QHeaderView,
    QMessageBox, QSpinBox, QHBoxLayout, QStackedWidget, QFrame
)
from PyQt6.QtGui import QIcon, QStandardItemModel, QStandardItem
from PyQt6.QtCore import pyqtSignal, Qt, QEvent
from ui.widgets.parameter_tree_view import ParameterSelectionTreeView
import qtawesome as qta

# Conditional imports for graceful degradation
try:
    from ui.workers.sensitivity_analysis_worker import SensitivityAnalysisWorker
    from analysis.sensitivity_analyzer import SensitivityAnalyzer
except ImportError as e:
    logging.critical(f"RedesignedSensitivityWidget: Failed to import core components: {e}.")
    SensitivityAnalysisWorker = None
    SensitivityAnalyzer = None

try:
    from PyQt6.QtWebEngineWidgets import QWebEngineView
except ImportError as e:
    logging.warning(f"RedesignedSensitivityWidget: PyQt6-WebEngineWidgets not found: {e}. Plots disabled.")
    QWebEngineView = None

logger = logging.getLogger(__name__)

class Card(QFrame):
    """A simple card widget for modern UI."""
    def __init__(self, title: str, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("card")
        main_layout = QVBoxLayout(self)
        
        self.title_label = QLabel(title)
        self.title_label.setObjectName("cardTitle")
        main_layout.addWidget(self.title_label)
        
        self.content_widget = QWidget()
        main_layout.addWidget(self.content_widget)

        # Basic styling
        self.setStyleSheet("""
            #card {
                border: 1px solid #E0E0E0;
                border-radius: 8px;
                background-color: #FFFFFF;
            }
            #cardTitle {
                font-size: 16px;
                font-weight: bold;
                padding: 10px;
                border-bottom: 1px solid #E0E0E0;
            }
        """)

class RedesignedSensitivityWidget(QWidget):
    """A redesigned, modern UI for Sensitivity Analysis."""
    help_requested = pyqtSignal(str)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.analyzer: Optional[SensitivityAnalyzer] = None
        self.worker: Optional[SensitivityAnalysisWorker] = None
        self.analysis_type: Optional[str] = None
        self.last_run_data: Optional[Dict[str, Any]] = None
        
        self._setup_ui()
        self._connect_signals()
        self._initialize_state()
        self.retranslateUi()

    def _setup_ui(self):
        main_layout = QHBoxLayout(self)
        
        # --- Configuration Panel ---
        config_panel = QWidget()
        config_layout = QVBoxLayout(config_panel)
        config_panel.setFixedWidth(400)

        self.wizard = QStackedWidget()
        self.step1_type_selection = self._create_step1_type_selection()
        self.step2_param_config = self._create_step2_param_config()
        self.step3_run = self._create_step3_run()

        self.wizard.addWidget(self.step1_type_selection)
        self.wizard.addWidget(self.step2_param_config)
        self.wizard.addWidget(self.step3_run)
        
        config_layout.addWidget(self.wizard)

        # --- Results Panel ---
        results_panel = self._create_results_panel()

        # --- Main Splitter ---
        splitter = QSplitter(Qt.Orientation.Horizontal)
        config_panel.setMinimumWidth(320)
        config_panel.setMaximumWidth(500)
        results_panel.setMinimumWidth(400)
        splitter.addWidget(config_panel)
        splitter.addWidget(results_panel)
        # Use stretch factors for better resizing behavior
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([350, 750])  # Initial sizes only

        main_layout.addWidget(splitter)

    def _create_step1_type_selection(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(20)

        title_layout = QHBoxLayout()
        title = QLabel("New Sensitivity Analysis")
        title.setObjectName("mainTitle")
        title_layout.addWidget(title)
        title_layout.addStretch()
        help_button = QPushButton(qta.icon("fa5s.question-circle"), "")
        help_button.setFlat(True)
        help_button.clicked.connect(lambda: self.help_requested.emit("analysis.sensitivity"))
        title_layout.addWidget(help_button)
        layout.addLayout(title_layout)

        one_way_card = self._create_analysis_type_card(
            "One-Way Analysis",
            "Analyze the effect of individual parameters on the objective.",
            "one_way"
        )
        two_way_card = self._create_analysis_type_card(
            "Two-Way Analysis",
            "Analyze the interaction between two parameters.",
            "two_way"
        )
        re_opt_card = self._create_analysis_type_card(
            "Re-Optimization",
            "Re-run optimization for a range of parameter values.",
            "re_opt"
        )

        layout.addWidget(one_way_card)
        layout.addWidget(two_way_card)
        layout.addWidget(re_opt_card)
        layout.addStretch()
        
        return widget

    def _create_analysis_type_card(self, title: str, description: str, analysis_type: str) -> QPushButton:
        button = QPushButton()
        layout = QVBoxLayout(button)
        
        title_label = QLabel(title)
        title_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        
        desc_label = QLabel(description)
        desc_label.setWordWrap(True)
        
        layout.addWidget(title_label)
        layout.addWidget(desc_label)
        
        button.setMinimumHeight(100)
        button.setProperty("analysis_type", analysis_type)
        button.clicked.connect(self._on_analysis_type_selected)
        
        return button

    def _create_step2_param_config(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        self.param_config_stack = QStackedWidget()
        
        # One-Way / Re-Opt Config
        self.one_way_config_widget = self._create_one_way_re_opt_config_ui()
        
        # Two-Way Config
        self.two_way_config_widget = self._create_two_way_config_ui()

        self.param_config_stack.addWidget(self.one_way_config_widget)
        self.param_config_stack.addWidget(self.two_way_config_widget)
        
        layout.addWidget(self.param_config_stack)

        nav_layout = QHBoxLayout()
        back_button = QPushButton("Back")
        back_button.clicked.connect(self._go_back)
        next_button = QPushButton("Next")
        next_button.clicked.connect(lambda: self.wizard.setCurrentIndex(2))
        nav_layout.addWidget(back_button)
        nav_layout.addStretch()
        nav_layout.addWidget(next_button)
        layout.addLayout(nav_layout)
        
        return widget

    def _create_one_way_re_opt_config_ui(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        param_card = Card("Select Parameters")
        param_card.content_widget.setLayout(QVBoxLayout())
        self.param_tree = ParameterSelectionTreeView()
        param_card.content_widget.layout().addWidget(self.param_tree)
        
        config_card = Card("Configure Analysis")
        grid = QGridLayout()
        config_card.content_widget.setLayout(grid)
        grid.addWidget(QLabel("Number of Steps:"), 0, 0)
        self.num_steps_spin = QSpinBox()
        self.num_steps_spin.setRange(3, 101)
        self.num_steps_spin.setValue(21)
        grid.addWidget(self.num_steps_spin, 0, 1)
        
        grid.addWidget(QLabel("Variation:"), 1, 0)
        self.variation_edit = QLineEdit()
        grid.addWidget(self.variation_edit, 1, 1)
        
        layout.addWidget(param_card)
        layout.addWidget(config_card)
        
        return widget

    def _create_two_way_config_ui(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        p1_card = Card("Select Parameter 1")
        p1_card.content_widget.setLayout(QVBoxLayout())
        self.p1_tree = ParameterSelectionTreeView()
        p1_card.content_widget.layout().addWidget(self.p1_tree)
        
        p1_config_card = Card("Configure Parameter 1")
        grid1 = QGridLayout()
        p1_config_card.content_widget.setLayout(grid1)
        grid1.addWidget(QLabel("Values:"), 0, 0)
        self.p1_values_edit = QLineEdit()
        grid1.addWidget(self.p1_values_edit, 0, 1)
        
        p2_card = Card("Select Parameter 2")
        p2_card.content_widget.setLayout(QVBoxLayout())
        self.p2_tree = ParameterSelectionTreeView()
        p2_card.content_widget.layout().addWidget(self.p2_tree)
        
        p2_config_card = Card("Configure Parameter 2")
        grid2 = QGridLayout()
        p2_config_card.content_widget.setLayout(grid2)
        grid2.addWidget(QLabel("Values:"), 0, 0)
        self.p2_values_edit = QLineEdit()
        grid2.addWidget(self.p2_values_edit, 0, 1)
        
        layout.addWidget(p1_card)
        layout.addWidget(p1_config_card)
        layout.addWidget(p2_card)
        layout.addWidget(p2_config_card)
        
        return widget

    def _create_step3_run(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        objective_card = Card("Select Objective")
        grid = QGridLayout()
        objective_card.content_widget.setLayout(grid)
        grid.addWidget(QLabel("Objective to Plot:"), 0, 0)
        self.objective_combo = QComboBox()
        self.objective_combo.addItems(['npv', 'recovery_factor', 'co2_utilization'])
        grid.addWidget(self.objective_combo, 0, 1)
        
        run_card = Card("Execute")
        run_card.content_widget.setLayout(QVBoxLayout())
        self.run_button = QPushButton("Confirm and Run Analysis")
        self.run_button.setIcon(QIcon.fromTheme("system-run"))
        run_card.content_widget.layout().addWidget(self.run_button)

        self.add_to_report_button = QPushButton("Add to Report")
        self.add_to_report_button.setIcon(QIcon.fromTheme("document-save"))
        self.add_to_report_button.setEnabled(False)
        run_card.content_widget.layout().addWidget(self.add_to_report_button)
        
        back_button = QPushButton("Back")
        back_button.clicked.connect(self._go_back)
        
        layout.addWidget(objective_card)
        layout.addWidget(run_card)
        layout.addStretch()
        layout.addWidget(back_button)
        
        return widget

    def _create_results_panel(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        self.results_tabs = QTabWidget()
        
        if QWebEngineView:
            self.plot_view_1 = QWebEngineView()
            self.plot_view_2 = QWebEngineView()
            self.surface_3d_view = QWebEngineView()
        else:
            self.plot_view_1 = QLabel("Plotting requires PyQt6-WebEngine.")
            self.plot_view_2 = QLabel("Plotting requires PyQt6-WebEngine.")
            self.surface_3d_view = QLabel("Plotting requires PyQt6-WebEngine.")
        
        self.data_table = QTableWidget()
        
        self.results_tabs.addTab(self.plot_view_1, "Tornado Plot")
        self.results_tabs.addTab(self.plot_view_2, "Spider Plot")
        self.results_tabs.addTab(self.surface_3d_view, "3D Surface")
        self.results_tabs.addTab(self.data_table, "Data")
        
        layout.addWidget(self.results_tabs)
        return container

    def _connect_signals(self):
        self.run_button.clicked.connect(self._run_analysis)
        self.add_to_report_button.clicked.connect(self._add_to_report)

    def _initialize_state(self):
        self.wizard.setCurrentIndex(0)
        if not SensitivityAnalyzer or not SensitivityAnalysisWorker:
            self.setEnabled(False)
            QMessageBox.critical(self, "Component Error", "Core components for SA are missing.")

    def retranslateUi(self):
        # Handle language changes
        pass

    def _on_analysis_type_selected(self):
        sender = self.sender()
        self.analysis_type = sender.property("analysis_type")
        
        if self.analysis_type in ["one_way", "re_opt"]:
            self.param_config_stack.setCurrentWidget(self.one_way_config_widget)
        elif self.analysis_type == "two_way":
            self.param_config_stack.setCurrentWidget(self.two_way_config_widget)
            
        self.wizard.setCurrentIndex(1)

    def _go_back(self):
        current_index = self.wizard.currentIndex()
        if current_index > 0:
            self.wizard.setCurrentIndex(current_index - 1)

    def update_analyzer(self, analyzer: Optional[SensitivityAnalyzer]):
        self.analyzer = analyzer
        if self.analyzer:
            try:
                param_structure = self.analyzer.get_configurable_parameters()
                self.param_tree.populate(param_structure)
                self.p1_tree.populate(param_structure)
                self.p2_tree.populate(param_structure)
                logger.info("RedesignedSensitivityWidget analyzer updated and parameters populated.")
            except Exception as e:
                logger.error(f"Failed to get configurable parameters from analyzer: {e}")
                QMessageBox.warning(self, self.tr("Analyzer Error"), self.tr("Could not load parameters: {}").format(e))
                self.param_tree.model.clear()
                self.p1_tree.model.clear()
                self.p2_tree.model.clear()
        else:
            self.param_tree.model.clear()
            self.p1_tree.model.clear()
            self.p2_tree.model.clear()
            logger.warning("RedesignedSensitivityWidget analyzer instance removed.")

    def _run_analysis(self):
        if not self.analyzer or not SensitivityAnalysisWorker:
            QMessageBox.critical(self, self.tr("Error"), self.tr("Analyzer components are not available."))
            return
        if self.worker and self.worker.isRunning():
            QMessageBox.warning(self, self.tr("Busy"), self.tr("An analysis is already in progress."))
            return

        if self.analysis_type == 'two_way':
            selected_items = self.p1_tree.get_selected_items() + self.p2_tree.get_selected_items()
        else:
            selected_items = self.param_tree.get_selected_items()

        objective = self.objective_combo.currentText()
        
        if not objective:
            QMessageBox.warning(self, self.tr("Input Error"), self.tr("Please select an objective/result for the analysis."))
            return
        if not selected_items:
            QMessageBox.warning(self, self.tr("Input Error"), self.tr("Please select at least one parameter for analysis."))
            return

        try:
            method_name, kwargs = self._prepare_run_args(self.analysis_type, selected_items, [objective])
            run_context = {"sa_type": self.analysis_type, "selected_items": selected_items, "objective": objective}
        except ValueError as e:
            QMessageBox.warning(self, self.tr("Configuration Error"), str(e))
            return

        self._clear_results()
        self.data_table.setColumnCount(1)
        self.data_table.setRowCount(1)
        self.data_table.setItem(0, 0, QTableWidgetItem(self.tr("<i>Running analysis, please wait...</i>")))
        self.run_button.setEnabled(False)
        self.add_to_report_button.setEnabled(False)

        self.worker = SensitivityAnalysisWorker(self.analyzer, method_name, kwargs, run_context)
        self.worker.result_ready.connect(self._on_result)
        self.worker.error_occurred.connect(self._on_error)
        self.worker.finished.connect(self._on_worker_finished)
        self.worker.start()
        logger.info(f"Sensitivity analysis started: Type='{self.analysis_type}', Method='{method_name}'")

    def _prepare_run_args(self, sa_type: str, items: List[Tuple[str,str]], objectives: List[str]) -> Tuple[str, Dict]:
        paths = [item[0] for item in items]
        if sa_type == "one_way":
            return "run_one_way_sensitivity", {
                "param_paths": paths, "variation_str": self.variation_edit.text(),
                "num_steps": self.num_steps_spin.value(), "objectives": objectives
            }
        elif sa_type == "two_way":
            if len(paths) != 2:
                raise ValueError(self.tr("Select exactly two parameters for Two-Way analysis."))
            val_str1, val_str2 = self.p1_values_edit.text(), self.p2_values_edit.text()
            if not val_str1 or not val_str2:
                raise ValueError(self.tr("Both parameter value fields must be filled for Two-Way analysis."))
            return "run_two_way_sensitivity", {
                "param1_path": paths[0], "param1_values_str": val_str1,
                "param2_path": paths[1], "param2_values_str": val_str2,
                "objectives": objectives
            }
        elif sa_type == "re_opt":
            if len(paths) != 1:
                raise ValueError(self.tr("Select exactly one parameter for Re-Optimization analysis."))
            return "run_reoptimization_sensitivity", {
                "primary_param_to_vary": paths[0], "variation_values_str": self.variation_edit.text(),
                "objectives_at_optimum": objectives
            }
        raise ValueError(self.tr("Analysis type '{sa_type}' is not implemented.").format(sa_type=sa_type))

    def _clear_results(self):
        if isinstance(self.plot_view_1, QWebEngineView): self.plot_view_1.setHtml("")
        if isinstance(self.plot_view_2, QWebEngineView): self.plot_view_2.setHtml("")
        if isinstance(self.surface_3d_view, QWebEngineView): self.surface_3d_view.setHtml("")
        self.data_table.clear()
        self.data_table.setRowCount(0)
        self.data_table.setColumnCount(0)
        logger.debug("Cleared previous analysis results.")

    def _on_result(self, df: pd.DataFrame, context: Dict):
        self._clear_results()
        if df.empty:
            self.data_table.setRowCount(1)
            self.data_table.setItem(0, 0, QTableWidgetItem(self.tr("Analysis produced no results.")))
            logger.warning("Received empty DataFrame from sensitivity analysis worker.")
            return

        self.last_run_data = {"df": df, "context": context}
        self.add_to_report_button.setEnabled(True)

        self._populate_data_table(df)
        self._generate_plots(df, context)
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

    def _generate_plots(self, df: pd.DataFrame, context: Dict):
        if not QWebEngineView or not self.analyzer:
            return

        sa_type = context.get("sa_type")
        objective = context.get("objective")
        selected_items = context.get("selected_items", [])
        
        if not all([sa_type, objective, selected_items]):
            logger.error("Plot generation failed: context from worker is missing key information.")
            return

        if objective not in df.columns:
            msg = self.tr("Objective '{objective}' not found in results. Cannot generate plots.").format(objective=objective)
            logger.warning(msg)
            if isinstance(self.plot_view_1, QWebEngineView):
                self.plot_view_1.setHtml(f"<p style='color:red;'>{msg}</p>")
            return
        
        try:
            if sa_type == "one_way":
                tornado = self.analyzer.plot_tornado_chart(df, objective)
                spider = self.analyzer.plot_spider_chart(df, objective)
                if tornado and isinstance(self.plot_view_1, QWebEngineView):
                    self.plot_view_1.setHtml(tornado.to_html(include_plotlyjs='cdn'))
                if spider and isinstance(self.plot_view_2, QWebEngineView):
                    self.plot_view_2.setHtml(spider.to_html(include_plotlyjs='cdn'))
            elif sa_type == "two_way":
                p1_path, p2_path = selected_items[0][0], selected_items[1][0]
                surface = self.analyzer.plot_3d_surface(df, p1_path, p2_path, objective)
                if surface and isinstance(self.surface_3d_view, QWebEngineView):
                    self.surface_3d_view.setHtml(surface.to_html(include_plotlyjs='cdn'))
            elif sa_type == "re_opt":
                reopt_fig = self._plot_reoptimization_chart(df, selected_items[0][0], objective)
                if reopt_fig and isinstance(self.plot_view_1, QWebEngineView):
                    self.plot_view_1.setHtml(reopt_fig.to_html(include_plotlyjs='cdn'))
        except Exception as e:
            logger.error(f"Error generating plot for {sa_type}: {e}", exc_info=True)
            if isinstance(self.plot_view_1, QWebEngineView):
                error_msg = self.tr("Plotting Error: {}").format(e)
                self.plot_view_1.setHtml(f"<p style='color:red;'>{error_msg}</p>")

    def _plot_reoptimization_chart(self, df: pd.DataFrame, varied_param: str, objective_col: str) -> Optional[go.Figure]:
        """Creates a multi-axis plot for re-optimization results."""
        opt_cols = [col for col in df.columns if col.startswith('opt_')]
        if not opt_cols: return None

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(x=df[varied_param], y=df[objective_col], name=objective_col.replace('_', ' ').title()), secondary_y=False)
        for col in opt_cols:
            fig.add_trace(go.Scatter(x=df[varied_param], y=df[col], name=col.replace('_', ' ').title()), secondary_y=True)

        fig.update_layout(
            title_text=self.tr("Re-Optimization Results for '{varied_param}'").format(varied_param=varied_param),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        fig.update_xaxes(title_text=varied_param)
        primary_y_title = self.tr("<b>Primary</b>: {}").format(objective_col.replace('_', ' ').title())
        secondary_y_title = self.tr("<b>Secondary</b>: Optimal Values")
        fig.update_yaxes(title_text=primary_y_title, secondary_y=False)
        fig.update_yaxes(title_text=secondary_y_title, secondary_y=True)
        return fig

    def _on_error(self, error_msg: str):
        self._clear_results()
        self.data_table.setColumnCount(1)
        self.data_table.setRowCount(1)
        self.data_table.setItem(0, 0, QTableWidgetItem(self.tr("Error: {}").format(error_msg)))
        QMessageBox.critical(self, self.tr("Analysis Error"), self.tr("An error occurred: {}").format(error_msg))
        logger.error(f"Sensitivity analysis worker reported an error: {error_msg}")

    def _on_worker_finished(self):
        self.run_button.setEnabled(self.analyzer is not None)
        if self.worker:
            self.worker.deleteLater() # Ensure proper cleanup
            self.worker = None
        logger.info("Sensitivity analysis worker finished.")

    def _add_to_report(self):
        if self.analyzer and self.last_run_data:
            if not hasattr(self.analyzer, 'report_runs'):
                self.analyzer.report_runs = []
            self.analyzer.report_runs.append(self.last_run_data)
            QMessageBox.information(self, "Success", "Analysis results added to the report.")
            self.add_to_report_button.setEnabled(False)
