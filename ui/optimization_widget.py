import logging
import json
import csv
import io
from datetime import datetime
from pathlib import Path
from typing import Optional, Any, Dict, List
from dataclasses import fields, is_dataclass, asdict
from functools import partial

import numpy as np
import pandas as pd
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGroupBox,
    QLabel,
    QComboBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QApplication,
    QSplitter,
    QAbstractItemView,
    QMessageBox,
    QSpinBox,
    QDoubleSpinBox,
    QFormLayout,
    QSizePolicy,
    QTabWidget,
    QGridLayout,
    QTextEdit,
    QDialog,
    QCheckBox,
    QDialogButtonBox,
    QTreeWidget,
    QTreeWidgetItem,
    QFileDialog,
    QListWidget,
    QListWidgetItem,
    QLineEdit,
)
from PyQt6.QtGui import QIcon, QFont
from PyQt6.QtCore import Qt, pyqtSignal, QObject, pyqtSlot, QEvent
from PyQt6.QtWebEngineWidgets import QWebEngineView
import plotly.graph_objects as go

from ui.qt_log_handler import QtLogHandler

from ui.workers.optimization_worker import OptimizationWorker
from core.optimisation_engine import OptimizationEngine
from core.data_models import (
    GeneticAlgorithmParams,
    EconomicParameters,
    BayesianOptimizationParams,
    EORParameters,
    ParticleSwarmParams,
    DifferentialEvolutionParams,
)
from core.Phys_engine_full.material_balance import create_material_balance_from_optimization

# --- Imports for integrated MMP Analysis ---
try:
    from ui.workers.well_analysis_worker import WellAnalysisWorker
    from core.data_models import WellData, PVTProperties
    from evaluation.mmp import MMP_METHODS

    _mmp_import_failed = False
except ImportError as e:
    logging.critical(f"OptimizationWidget: Failed to import MMP analysis components: {e}")

    class WellData:
        pass

    class PVTProperties:
        pass

    class WellAnalysisWorker:
        pass

    MMP_METHODS = {}
    _mmp_import_failed = True


from config_manager import ConfigManager

logger = logging.getLogger(__name__)


class OptimizationWidget(QWidget):
    """Main widget for configuring, running, and analyzing optimization tasks."""

    optimization_completed = pyqtSignal(dict)
    open_configuration_requested = pyqtSignal()
    # Signal to emit the single, representative MMP value for the engine
    representative_mmp_calculated = pyqtSignal(float)
    # Signal to request engine type change (emits engine_type: str)
    engine_type_change_requested = pyqtSignal(str)

    def __init__(self, config_manager: ConfigManager, parent: Optional[QWidget] = None):
        super().__init__(parent)
        # Debug: Check what we received
        logger.debug(f"Received config_manager: {type(config_manager)}")
        logger.debug(f"Received parent: {type(parent)}")
        self.config_manager = config_manager
        ui_config = self.config_manager.get_section("ui_config").get("optimization", {})
        self.OPTIMIZATION_METHODS = ui_config.get("methods", {})
        self.OPTIMIZATION_OBJECTIVES = ui_config.get("objectives", {})
        self.PARAMETER_METADATA = ui_config.get("parameter_metadata", {})

        self.engine: Optional[OptimizationEngine] = None
        self.worker: Optional[OptimizationWorker] = None
        self.current_results: Optional[Dict[str, Any]] = None
        self.selected_engine_type: Optional[str] = None

        # --- MMP Analysis State ---
        self.well_data_list: List[WellData] = []
        self.pvt_data: Optional[PVTProperties] = None
        self.mmp_worker: Optional[WellAnalysisWorker] = None

        self.ga_param_inputs: Dict[str, QWidget] = {}
        self.bo_param_inputs: Dict[str, QWidget] = {}
        self.pso_param_inputs: Dict[str, QWidget] = {}
        self.de_param_inputs: Dict[str, QWidget] = {}
        self.convergence_live_data: List[Dict[str, float]] = []

        self.linked_min_max_widgets: Dict[str, Dict[str, QWidget]] = {}

        self._initial_setup_complete = False
        self._setup_logging()
        self._setup_ui()
        self._connect_signals()

        if not all(
            [OptimizationWorker, OptimizationEngine, GeneticAlgorithmParams, EconomicParameters]
        ):
            self.setEnabled(False)
            QMessageBox.critical(
                self,
                self.tr("Component Error"),
                self.tr("Core components for optimization are missing."),
            )

        if _mmp_import_failed:
            if hasattr(self, "mmp_tab_widget"):
                self.mmp_tab_widget.setEnabled(False)
                self.mmp_tab_widget.setToolTip(
                    self.tr("MMP analysis components failed to load. This feature is disabled.")
                )

        self._initial_setup_complete = True
        self.retranslateUi()

    def changeEvent(self, event: QEvent):
        """Handles events sent to the widget, specifically for language changes."""
        if event.type() == QEvent.Type.LanguageChange:
            self.retranslateUi()
        super().changeEvent(event)

    def retranslateUi(self):
        """Updates all user-visible text in the widget to the current language."""
        # Main Panels
        self.log_group.setTitle(self.tr("Optimization Engine Log"))
        self.status_label.setText(self.tr("<i>Ready to start optimization.</i>"))

        # Config Panel Tabs
        self.config_tabs.setTabText(
            self.config_tabs.indexOf(self.mmp_tab_widget), self.tr("1. MMP Calculation")
        )
        self.config_tabs.setTabText(
            self.config_tabs.indexOf(self.setup_tab_widget), self.tr("2. Optimization Setup")
        )
        self.config_tabs.setTabText(
            self.config_tabs.indexOf(self.hyperparams_widget), self.tr("3. Hyperparameters")
        )

        # Core Setup Group
        self.setup_group.setTitle(self.tr("Core Setup"))
        self.method_label.setText(self.tr("Method:"))
        self.objective_label.setText(self.tr("Objective:"))
        self.resolution_label.setText(self.tr("Resolution:"))

        # Populate method combo box with translated text
        self.method_combo.blockSignals(True)
        current_data = self.method_combo.currentData()
        self.method_combo.clear()
        for text, data in self.OPTIMIZATION_METHODS.items():
            self.method_combo.addItem(self.tr(text), userData=data)
        idx = self.method_combo.findData(current_data)
        if idx != -1:
            self.method_combo.setCurrentIndex(idx)
        self.method_combo.blockSignals(False)

        # Populate objective combo box
        self.objective_combo.blockSignals(True)
        current_data = self.objective_combo.currentData()
        self.objective_combo.clear()
        for text, data in self.OPTIMIZATION_OBJECTIVES.items():
            self.objective_combo.addItem(self.tr(text), userData=data)
        idx = self.objective_combo.findData(current_data)
        if idx != -1:
            self.objective_combo.setCurrentIndex(idx)
        self.objective_combo.blockSignals(False)

        # Populate resolution combo box
        self.resolution_combo.blockSignals(True)
        current_res = self.resolution_combo.currentText().lower()
        self.resolution_combo.clear()
        resolutions = ["Yearly", "Quarterly", "Monthly", "Weekly"]
        for res in resolutions:
            self.resolution_combo.addItem(self.tr(res), userData=res.lower())
        idx = self.resolution_combo.findData(current_res)
        if idx != -1:
            self.resolution_combo.setCurrentIndex(idx)
        self.resolution_combo.blockSignals(False)

        # Action Buttons
        self.configure_button.setText(self.tr("Configure Parameters..."))
        self.configure_button.setToolTip(
            self.tr(
                "Open the main configuration panel for economic, operational, and other settings."
            )
        )
        self.run_button.setText(self.tr("Run Optimization"))
        self.run_button.setToolTip(
            self.tr("Load a project to enable the optimization engine.")
            if not self.run_button.isEnabled()
            else self.tr("Run the configured optimization.")
        )

        # MMP Panel
        self.mmp_config_group.setTitle(self.tr("MMP Analysis Configuration"))
        self.mmp_well_select_label.setText(self.tr("Select Well:"))
        self.mmp_method_label.setText(self.tr("MMP Method:"))
        self.mmp_method_inputs_group.setTitle(self.tr("Method-Specific Inputs"))
        self.mmp_c7_label.setText(self.tr("C7+ Molecular Weight (g/mol):"))
        self.mmp_c7_mw_input.setToolTip(
            self.tr("Used for the 'Hybrid GH' and 'Alston' correlations.")
        )
        self.mmp_gas_comp_label.setText(self.tr("<b>Gas Composition (Mole Fraction)</b>"))
        self.mmp_co2_label.setText(self.tr("CO₂:"))
        self.mmp_ch4_label.setText(self.tr("CH₄ (Methane):"))
        self.mmp_n2_label.setText(self.tr("N₂ (Nitrogen):"))
        self.mmp_n2_comp_input.setToolTip(self.tr("Used for the 'Alston' correlation."))
        self.mmp_normalize_btn.setText(self.tr("Normalize to 1.0"))
        self.mmp_calculate_button.setText(self.tr(" Calculate MMP Profile"))

        # Hyperparameters Panel
        self.ga_params_group.setTitle(self.tr("Genetic Algorithm (PyGAD)"))
        self.bo_params_group.setTitle(self.tr("Bayesian Optimization"))
        self.pso_params_group.setTitle(self.tr("Particle Swarm Optimization"))
        self.de_params_group.setTitle(self.tr("Differential Evolution"))
        self.hyperparam_tabs.setTabText(
            self.hyperparam_tabs.indexOf(self.ga_params_group), self.tr("GA")
        )
        self.hyperparam_tabs.setTabText(
            self.hyperparam_tabs.indexOf(self.bo_params_group), self.tr("BO")
        )
        self.hyperparam_tabs.setTabText(
            self.hyperparam_tabs.indexOf(self.pso_params_group), self.tr("PSO")
        )
        self.hyperparam_tabs.setTabText(
            self.hyperparam_tabs.indexOf(self.de_params_group), self.tr("DE")
        )

        # Results Panel Tabs
        self.results_tabs.setTabText(
            self.results_tabs.indexOf(self.input_summary_tab), self.tr("Input Parameters")
        )
        self.results_tabs.setTabText(
            self.results_tabs.indexOf(self.mmp_results_tab), self.tr("MMP Analysis")
        )
        self.results_tabs.setTabText(
            self.results_tabs.indexOf(self.summary_tab), self.tr("Optimization Summary")
        )
        self.results_tabs.setTabText(
            self.results_tabs.indexOf(self.detailed_summary_tab), self.tr("Detailed Summary")
        )
        self.results_tabs.setTabText(
            self.results_tabs.indexOf(self.analysis_tab), self.tr("Optimization Analysis")
        )

        # MMP Results Panel
        self.mmp_results_group.setTitle(self.tr("MMP Analysis Results"))
        self.mmp_results_table.setHorizontalHeaderLabels(
            [
                self.tr("Depth (ft)"),
                self.tr("Temperature (°F)"),
                self.tr("Oil Gravity (°API)"),
                self.tr("MMP (psia)"),
            ]
        )
        self.mmp_status_label.setText(self.tr("<i>Load project data to perform analysis.</i>"))

        # Input Summary Panel
        self.input_params_tree.setHeaderLabels([self.tr("Parameter"), self.tr("Value")])
        self._update_input_summary_tab()  # Refresh with translated names

        # Optimization Summary Panel
        self.summary_label.setText(self.tr("<i>Run optimization to view results.</i>"))
        self.export_button.setText(self.tr("Export Run Data..."))
        self.export_button.setToolTip(
            self.tr("Export all input parameters and optimization results to a file.")
        )
        self.results_table.setHorizontalHeaderLabels(
            [self.tr("Parameter"), self.tr("Optimized Value"), self.tr("Description")]
        )

        # Analysis Panel
        self.analysis_plot_label.setText(self.tr("<b>Available Plots</b>"))
        self.analysis_param_label.setText(self.tr("Parameter:"))

        # Repopulate plot list with translated names
        current_text = self.plot_list.currentItem().text() if self.plot_list.currentItem() else None
        self.plot_list.clear()
        plot_options = ["Convergence", "Final Production Profiles", "Parameter Sensitivity"]
        if self.current_results and "bayes_opt_obj" in self.current_results:
            plot_options.append("Objective vs. Parameter (BO)")

        new_current_item = None
        for option in plot_options:
            translated_option = self.tr(option)
            item = QListWidgetItem(QIcon.fromTheme("view-plot"), translated_option)
            self.plot_list.addItem(item)
            if translated_option == current_text:
                new_current_item = item
        if new_current_item:
            self.plot_list.setCurrentItem(new_current_item)

    def _setup_logging(self):
        """Configures the custom log handler to route log messages to the UI."""
        self.log_handler = QtLogHandler()
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S"
        )
        self.log_handler.setFormatter(formatter)
        logging.getLogger("co2eor_optimizer").addHandler(self.log_handler)

    def _setup_ui(self):
        """Creates the main layout and panels of the widget."""
        main_layout = QVBoxLayout(self)
        v_splitter = QSplitter(Qt.Orientation.Vertical)

        h_splitter = QSplitter(Qt.Orientation.Horizontal)
        config_panel = self._create_config_panel()
        config_panel.setMinimumWidth(350)
        results_panel = self._create_results_panel()
        results_panel.setMinimumWidth(400)
        h_splitter.addWidget(config_panel)
        h_splitter.addWidget(results_panel)
        # Use stretch factors for better resizing behavior
        h_splitter.setStretchFactor(0, 0)
        h_splitter.setStretchFactor(1, 1)
        h_splitter.setSizes([400, 800])  # Initial sizes only

        self.log_group = QGroupBox()
        self.log_group.setMinimumHeight(120)
        self.log_group.setMaximumHeight(300)
        log_layout = QVBoxLayout(self.log_group)
        self.log_display = QTextEdit()
        self.log_display.setReadOnly(True)
        font = QFont("Monospace")
        font.setPointSize(QApplication.font().pointSize())
        self.log_display.setFont(font)
        log_layout.addWidget(self.log_display)

        v_splitter.addWidget(h_splitter)
        v_splitter.addWidget(self.log_group)
        # Use stretch factors for better resizing behavior
        v_splitter.setStretchFactor(0, 1)
        v_splitter.setStretchFactor(1, 0)
        v_splitter.setSizes([600, 150])  # Initial sizes only

        main_layout.addWidget(v_splitter, 1)

        self.status_label = QLabel()
        self.status_label.setWordWrap(True)
        main_layout.addWidget(self.status_label)

    def _create_config_panel(self) -> QWidget:
        """Creates the left-hand panel for optimization setup and configuration."""
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)

        self.config_tabs = QTabWidget()

        # --- MMP Calculation Tab ---
        self.mmp_tab_widget = self._create_mmp_panel()
        self.config_tabs.addTab(self.mmp_tab_widget, "")

        # --- Core Setup Tab ---
        self.setup_tab_widget = QWidget()
        setup_layout = QVBoxLayout(self.setup_tab_widget)
        self.setup_group = QGroupBox()
        setup_grid = QGridLayout(self.setup_group)
        setup_grid.setColumnStretch(0, 0)
        setup_grid.setColumnStretch(1, 1)
        self.method_combo = QComboBox()
        self.method_combo.setMinimumWidth(150)
        self.objective_combo = QComboBox()
        self.objective_combo.setMinimumWidth(150)
        self.resolution_combo = QComboBox()
        self.resolution_combo.setMinimumWidth(150)
        self.method_label = QLabel()
        self.method_label.setMinimumWidth(100)
        self.objective_label = QLabel()
        self.objective_label.setMinimumWidth(100)
        self.resolution_label = QLabel()
        self.resolution_label.setMinimumWidth(100)
        setup_grid.addWidget(self.method_label, 0, 0)
        setup_grid.addWidget(self.method_combo, 0, 1)
        setup_grid.addWidget(self.objective_label, 1, 0)
        setup_grid.addWidget(self.objective_combo, 1, 1)
        setup_grid.addWidget(self.resolution_label, 2, 0)
        setup_grid.addWidget(self.resolution_combo, 2, 1)
        setup_layout.addWidget(self.setup_group)
        # Note: Engine selection moved to Config Widget (single source of truth)
        setup_layout.addStretch()
        self.config_tabs.addTab(self.setup_tab_widget, "")

        # --- Hyperparameters Tab (with sub-tabs) ---
        self.hyperparams_widget = self._create_hyperparameters_panel()
        self.config_tabs.addTab(self.hyperparams_widget, "")

        layout.addWidget(self.config_tabs)

        # --- Action Buttons ---
        action_button_layout = QHBoxLayout()
        self.configure_button = QPushButton()
        self.configure_button.setIcon(QIcon.fromTheme("document-properties"))
        self.run_button = QPushButton()
        self.run_button.setIcon(QIcon.fromTheme("system-run"))
        self.run_button.setEnabled(False)

        action_button_layout.addWidget(self.configure_button)
        action_button_layout.addStretch()
        action_button_layout.addWidget(self.run_button)
        layout.addLayout(action_button_layout)

        return container

    def _create_mmp_panel(self) -> QWidget:
        """Creates the panel for MMP calculation, integrated into the config area."""
        container = QWidget()
        layout = QVBoxLayout(container)

        self.mmp_config_group = QGroupBox()
        form_layout = QFormLayout(self.mmp_config_group)
        # Set consistent label widths to prevent squishing
        form_layout.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        self.mmp_well_select_combo = QComboBox()
        self.mmp_well_select_combo.setMinimumWidth(150)
        self.mmp_well_select_label = QLabel()
        self.mmp_well_select_label.setMinimumWidth(100)
        form_layout.addRow(self.mmp_well_select_label, self.mmp_well_select_combo)
        self.mmp_method_combo = QComboBox()
        self.mmp_method_combo.setMinimumWidth(150)
        self.mmp_method_combo.addItem(self.tr("Auto-Select"), "auto")
        for name, func in MMP_METHODS.items():
            self.mmp_method_combo.addItem(name.replace("_", " ").title(), name)
        self.mmp_method_label = QLabel()
        self.mmp_method_label.setMinimumWidth(100)
        form_layout.addRow(self.mmp_method_label, self.mmp_method_combo)
        layout.addWidget(self.mmp_config_group)

        self.mmp_method_inputs_group = QGroupBox()
        method_inputs_layout = QVBoxLayout(self.mmp_method_inputs_group)

        self.mmp_c7_mw_widget = QWidget()
        c7_layout = QHBoxLayout(self.mmp_c7_mw_widget)
        c7_layout.setContentsMargins(0, 0, 0, 0)
        self.mmp_c7_label = QLabel()
        self.mmp_c7_label.setMinimumWidth(80)
        c7_layout.addWidget(self.mmp_c7_label)
        self.mmp_c7_mw_input = QDoubleSpinBox()
        self.mmp_c7_mw_input.setRange(50.0, 250.0)
        self.mmp_c7_mw_input.setValue(190.0)
        c7_layout.addWidget(self.mmp_c7_mw_input, 1)
        method_inputs_layout.addWidget(self.mmp_c7_mw_widget)

        self.mmp_gas_comp_widget = QWidget()
        gas_layout = QGridLayout(self.mmp_gas_comp_widget)
        gas_layout.setContentsMargins(0, 0, 0, 0)
        gas_layout.setColumnStretch(0, 0)
        gas_layout.setColumnStretch(1, 1)
        self.mmp_gas_comp_label = QLabel()
        gas_layout.addWidget(self.mmp_gas_comp_label, 0, 0, 1, 2)
        self.mmp_co2_label = QLabel()
        self.mmp_co2_label.setMinimumWidth(80)
        gas_layout.addWidget(self.mmp_co2_label, 1, 0)
        self.mmp_co2_comp_input = QDoubleSpinBox()
        self.mmp_co2_comp_input.setRange(0.0, 1.0)
        self.mmp_co2_comp_input.setValue(1.0)
        self.mmp_co2_comp_input.setDecimals(3)
        gas_layout.addWidget(self.mmp_co2_comp_input, 1, 1)
        self.mmp_ch4_label = QLabel()
        self.mmp_ch4_label.setMinimumWidth(80)
        gas_layout.addWidget(self.mmp_ch4_label, 2, 0)
        self.mmp_ch4_comp_input = QDoubleSpinBox()
        self.mmp_ch4_comp_input.setRange(0.0, 1.0)
        self.mmp_ch4_comp_input.setValue(0.0)
        self.mmp_ch4_comp_input.setDecimals(3)
        gas_layout.addWidget(self.mmp_ch4_comp_input, 2, 1)

        self.mmp_n2_label = QLabel()
        self.mmp_n2_label.setMinimumWidth(80)
        gas_layout.addWidget(self.mmp_n2_label, 3, 0)
        self.mmp_n2_comp_input = QDoubleSpinBox()
        self.mmp_n2_comp_input.setRange(0.0, 1.0)
        self.mmp_n2_comp_input.setValue(0.0)
        self.mmp_n2_comp_input.setDecimals(3)
        gas_layout.addWidget(self.mmp_n2_comp_input, 3, 1)

        self.mmp_normalize_btn = QPushButton()
        self.mmp_normalize_btn.clicked.connect(self._mmp_normalize_gas_composition)
        gas_layout.addWidget(self.mmp_normalize_btn, 4, 0, 1, 2)

        method_inputs_layout.addWidget(self.mmp_gas_comp_widget)

        self.mmp_method_inputs_group.setVisible(False)
        layout.addWidget(self.mmp_method_inputs_group)

        layout.addStretch()

        self.mmp_calculate_button = QPushButton()
        self.mmp_calculate_button.setIcon(QIcon.fromTheme("view-plot"))
        layout.addWidget(self.mmp_calculate_button)
        return container

    def _create_hyperparameters_panel(self) -> QWidget:
        """Creates the nested tab widget for optimizer-specific hyperparameters."""
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(5, 5, 5, 5)
        self.hyperparam_tabs = QTabWidget()

        # GA Tab
        self.ga_params_group = QGroupBox()
        self.ga_params_form = QFormLayout(self.ga_params_group)
        self.hyperparam_tabs.addTab(self.ga_params_group, "")

        # BO Tab
        self.bo_params_group = QGroupBox()
        self.bo_params_form = QFormLayout(self.bo_params_group)
        self.hyperparam_tabs.addTab(self.bo_params_group, "")

        # PSO Tab
        self.pso_params_group = QGroupBox()
        self.pso_params_form = QFormLayout(self.pso_params_group)
        self.hyperparam_tabs.addTab(self.pso_params_group, "")

        # DE Tab
        self.de_params_group = QGroupBox()
        self.de_params_form = QFormLayout(self.de_params_group)
        self.hyperparam_tabs.addTab(self.de_params_group, "")

        layout.addWidget(self.hyperparam_tabs)
        return container

    def _create_results_panel(self) -> QWidget:
        """Creates the right-hand panel for displaying results and plots."""
        container = QWidget()
        layout = QVBoxLayout(container)
        self.results_tabs = QTabWidget()

        self.input_summary_tab = self._create_input_summary_panel()
        self.mmp_results_tab = self._create_mmp_results_panel()
        self.summary_tab = self._create_summary_results_panel()

        self.detailed_summary_tab = QWidget()
        detailed_layout = QVBoxLayout(self.detailed_summary_tab)
        self.detailed_summary_label = QLabel()
        detailed_layout.addWidget(self.detailed_summary_label)
        self.detailed_summary_table = QTableWidget()
        self.detailed_summary_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.detailed_summary_table.setAlternatingRowColors(True)
        detailed_layout.addWidget(self.detailed_summary_table)

        self.analysis_tab = self._create_analysis_panel()

        self.results_tabs.addTab(self.input_summary_tab, "")
        self.results_tabs.addTab(self.mmp_results_tab, "")
        self.results_tabs.addTab(self.summary_tab, "")
        self.results_tabs.addTab(self.detailed_summary_tab, "")
        self.results_tabs.addTab(self.analysis_tab, "")

        layout.addWidget(self.results_tabs)
        return container

    def _create_mmp_results_panel(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        self.mmp_results_group = QGroupBox()
        results_layout = QVBoxLayout(self.mmp_results_group)

        v_splitter = QSplitter(Qt.Orientation.Vertical)

        self.mmp_plot_view = QWebEngineView()
        self.mmp_plot_view.setMinimumHeight(300)
        v_splitter.addWidget(self.mmp_plot_view)

        self.mmp_results_table = QTableWidget()
        self.mmp_results_table.setColumnCount(4)
        self.mmp_results_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.mmp_results_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.mmp_results_table.setAlternatingRowColors(True)
        self.mmp_results_table.setMinimumHeight(150)
        v_splitter.addWidget(self.mmp_results_table)

        v_splitter.setSizes([400, 200])
        results_layout.addWidget(v_splitter, 1)

        self.mmp_status_label = QLabel()
        self.mmp_status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        results_layout.addWidget(self.mmp_status_label)
        layout.addWidget(self.mmp_results_group)
        return container

    def _create_input_summary_panel(self) -> QWidget:
        """Creates the panel that shows a tree view of all input parameters."""
        container = QWidget()
        layout = QVBoxLayout(container)
        self.input_params_tree = QTreeWidget()
        self.input_params_tree.header().setSectionResizeMode(
            0, QHeaderView.ResizeMode.ResizeToContents
        )
        self.input_params_tree.header().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.input_params_tree)
        return container

    def _create_summary_results_panel(self) -> QWidget:
        """Creates the main results panel with summary text and a table of optimized values."""
        container = QWidget()
        layout = QVBoxLayout(container)

        summary_header_layout = QHBoxLayout()
        self.summary_label = QLabel()
        self.summary_label.setWordWrap(True)
        summary_header_layout.addWidget(self.summary_label, 1)

        self.export_button = QPushButton()
        self.export_button.setIcon(QIcon.fromTheme("document-save"))
        self.export_button.setEnabled(False)
        summary_header_layout.addWidget(self.export_button)
        layout.addLayout(summary_header_layout)

        self.results_table = QTableWidget()
        self.results_table.setColumnCount(3)
        self.results_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.results_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.ResizeToContents
        )
        self.results_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.ResizeToContents
        )
        self.results_table.horizontalHeader().setSectionResizeMode(
            2, QHeaderView.ResizeMode.Stretch
        )
        layout.addWidget(self.results_table)
        return container

    def _create_analysis_panel(self) -> QWidget:
        """Creates the new, enhanced panel for plotting and visualization."""
        container = QWidget()
        layout = QHBoxLayout(container)
        splitter = QSplitter()

        # Left side: Plot selection list
        plot_selection_widget = QWidget()
        plot_selection_layout = QVBoxLayout(plot_selection_widget)
        self.analysis_plot_label = QLabel()
        plot_selection_layout.addWidget(self.analysis_plot_label)
        self.plot_list = QListWidget()
        self.plot_list.setFixedWidth(200)
        plot_selection_layout.addWidget(self.plot_list)
        splitter.addWidget(plot_selection_widget)

        # Right side: Plot view and dynamic controls
        plot_view_widget = QWidget()
        plot_view_layout = QVBoxLayout(plot_view_widget)

        self.plot_controls_widget = QWidget()
        self.plot_controls_layout = QGridLayout(self.plot_controls_widget)
        self.plot_controls_layout.setContentsMargins(0, 0, 0, 0)
        self.sensitivity_param_combo = QComboBox()
        self.bo_sensitivity_param_combo = QComboBox()
        self.analysis_param_label = QLabel()
        self.plot_controls_layout.addWidget(self.analysis_param_label, 0, 0)
        self.plot_controls_layout.addWidget(self.sensitivity_param_combo, 0, 1)
        self.plot_controls_layout.addWidget(self.bo_sensitivity_param_combo, 0, 1)
        self.plot_controls_layout.setColumnStretch(1, 1)
        plot_view_layout.addWidget(self.plot_controls_widget)

        self.analysis_plot_view = QWebEngineView()
        self.analysis_plot_view.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        plot_view_layout.addWidget(self.analysis_plot_view, 1)
        splitter.addWidget(plot_view_widget)

        layout.addWidget(splitter)
        splitter.setSizes([200, 650])
        return container

    def _connect_signals(self):
        """Connects all widget signals to their corresponding slots."""
        self.method_combo.currentTextChanged.connect(self._on_method_changed)
        self.run_button.clicked.connect(self._run_optimization)
        self.configure_button.clicked.connect(self.open_configuration_requested.emit)
        self.log_handler.emitter.log_record_received.connect(self._append_log_message)
        self.export_button.clicked.connect(self._export_run_data)

        # Plotting signals
        self.plot_list.currentItemChanged.connect(self._on_analysis_plot_selected)
        self.sensitivity_param_combo.currentTextChanged.connect(self._generate_selected_plot)
        self.bo_sensitivity_param_combo.currentTextChanged.connect(self._generate_selected_plot)

        # --- MMP Signals ---
        self.mmp_calculate_button.clicked.connect(self._run_mmp_calculation)
        self.mmp_well_select_combo.currentIndexChanged.connect(self._mmp_update_method_availability)
        self.mmp_method_combo.currentTextChanged.connect(
            self._mmp_update_method_specific_inputs_visibility
        )

    @pyqtSlot(logging.LogRecord)
    def _append_log_message(self, record: logging.LogRecord):
        """Appends a formatted message to the log display."""
        import datetime

        timestamp = datetime.datetime.fromtimestamp(record.created).strftime("%H:%M:%S")
        message = f"{timestamp} [{record.levelname}] {record.name}: {record.getMessage()}"
        self.log_display.append(message)

    def update_engine(self, engine: Optional[OptimizationEngine]):
        self.engine = engine
        is_engine_ready = self.engine is not None
        self.run_button.setEnabled(is_engine_ready)
        self.run_button.setToolTip(
            self.tr("Run the configured optimization.")
            if is_engine_ready
            else self.tr("Load a project to enable.")
        )

        if self.engine:
            self._populate_dataclass_form(
                self.engine.ga_params_default_config, self.ga_params_form, self.ga_param_inputs
            )
            self._populate_dataclass_form(
                self.engine.bo_params_default_config, self.bo_params_form, self.bo_param_inputs
            )
            self._populate_dataclass_form(
                self.engine.pso_params_default_config, self.pso_params_form, self.pso_param_inputs
            )
            self._populate_dataclass_form(
                self.engine.de_params_default_config, self.de_params_form, self.de_param_inputs
            )
            self._on_method_changed(self.method_combo.currentText())
            logger.info(
                "OptimizationWidget engine instance updated and all hyperparameter inputs displayed."
            )

            # Note: Engine selection widget moved to Config Widget
            # The Config Widget is now the single source of truth for engine selection
            current_engine_type = getattr(self.engine, "simulation_engine_type", None)
            if current_engine_type:
                logger.info(
                    f"Current engine type: {current_engine_type} (managed by Config Widget)"
                )
        else:
            self._clear_form(self.ga_params_form, self.ga_param_inputs)
            self._clear_form(self.bo_params_form, self.bo_param_inputs)
            self._clear_form(self.pso_params_form, self.pso_param_inputs)
            self._clear_form(self.de_params_form, self.de_param_inputs)
            self._update_input_summary_tab()
            logger.warning("OptimizationWidget engine instance removed.")

    def update_project_data(self, wells: List[WellData], pvt: Optional[PVTProperties]):
        """Receives well and PVT data from the main window to enable MMP analysis."""
        logger.info(f"OptimizationWidget updating MMP data with {len(wells)} wells.")
        self.well_data_list, self.pvt_data = wells, pvt
        self.mmp_well_select_combo.blockSignals(True)
        self.mmp_well_select_combo.clear()
        if not self.well_data_list:
            self.mmp_well_select_combo.addItem(self.tr("No wells loaded"))
            self.mmp_well_select_combo.setEnabled(False)
        else:
            for well in self.well_data_list:
                self.mmp_well_select_combo.addItem(well.name, userData=well)
            self.mmp_well_select_combo.setEnabled(True)
        self.mmp_well_select_combo.blockSignals(False)
        self._mmp_update_button_state()
        self._mmp_update_method_availability()

    @pyqtSlot(dict)
    def on_configurations_updated(self, full_config_data: Dict[str, Any]):
        """Slot to handle updates from the main ConfigWidget."""
        logger.info("OptimizationWidget received configuration update. Refreshing forms.")
        if not self._initial_setup_complete:
            return

        for dc_name, dc_instance in full_config_data.items():
            if dc_name == GeneticAlgorithmParams.__name__:
                self._populate_dataclass_form(
                    dc_instance, self.ga_params_form, self.ga_param_inputs
                )
            elif dc_name == BayesianOptimizationParams.__name__:
                self._populate_dataclass_form(
                    dc_instance, self.bo_params_form, self.bo_param_inputs
                )
            elif dc_name == ParticleSwarmParams.__name__:
                self._populate_dataclass_form(
                    dc_instance, self.pso_params_form, self.pso_param_inputs
                )
            elif dc_name == DifferentialEvolutionParams.__name__:
                self._populate_dataclass_form(
                    dc_instance, self.de_params_form, self.de_param_inputs
                )

        self._update_input_summary_tab()

    def _on_method_changed(self, method_name: str):
        """Shows/hides the relevant hyperparameter sub-tabs based on the selected method."""
        method_key = (self.method_combo.currentData() or "").lower()
        is_hybrid = "hybrid" in method_key

        self.hyperparam_tabs.setTabVisible(0, "genetic" in method_key or is_hybrid)  # GA
        self.hyperparam_tabs.setTabVisible(1, "bayesian" in method_key or is_hybrid)  # BO
        self.hyperparam_tabs.setTabVisible(2, "pso" in method_key)  # PSO
        self.hyperparam_tabs.setTabVisible(3, "de" in method_key)  # DE

        for i in range(self.hyperparam_tabs.count()):
            if self.hyperparam_tabs.isTabVisible(i):
                self.hyperparam_tabs.setCurrentIndex(i)
                break

        if self._initial_setup_complete:
            self._update_input_summary_tab()

    def _clear_form(self, form_layout: QFormLayout, input_dict: Dict):
        """Removes all rows from a QFormLayout and clears the input widget dictionary."""
        while form_layout.rowCount() > 0:
            form_layout.removeRow(0)
        input_dict.clear()

    def _populate_dataclass_form(
        self, dc_instance: Any, form_layout: QFormLayout, input_dict: Dict
    ):
        """Dynamically creates and populates a QFormLayout with widgets for a given dataclass instance."""
        self._clear_form(form_layout, input_dict)
        if not (dc_instance and is_dataclass(dc_instance)):
            return

        form_name = dc_instance.__class__.__name__
        self.linked_min_max_widgets[form_name] = {}

        min_max_pairs = [("keep_elitism", "sol_per_pop", "<")]

        for field in fields(dc_instance):
            meta = self.PARAMETER_METADATA.get(field.name, {})
            display_name = self.tr(meta.get("display", field.name.replace("_", " ").title()))
            unit = meta.get("unit", "")
            description = self.tr(meta.get("description", self.tr("No description available.")))
            label_text = f"{display_name} {unit}".strip()
            label_widget = QLabel(label_text)
            label_widget.setToolTip(description)

            value = getattr(dc_instance, field.name)
            widget = None

            if field.type is str:
                widget = QComboBox()
                options = {
                    "acquisition_function": ["ucb", "ei", "poi"],
                    "strategy": [
                        "best1bin",
                        "best1exp",
                        "rand1exp",
                        "randtobest1exp",
                        "currenttobest1exp",
                        "best2exp",
                        "rand2exp",
                        "randtobest1bin",
                        "currenttobest1bin",
                        "best2bin",
                        "rand2bin",
                        "rand1bin",
                    ],
                    "parent_selection_type": ["sss", "rws", "sus", "rank", "random", "tournament"],
                    "crossover_type": ["single_point", "two_points", "uniform", "scattered"],
                    "mutation_type": ["random", "swap", "inversion", "scramble", "adaptive"],
                    "constraint_handling_method": ["static", "adaptive", "death", "adaptive_penalty"],
                }
                widget.addItems(options.get(field.name, []))
                if value in [widget.itemText(i) for i in range(widget.count())]:
                    widget.setCurrentText(value)
            elif field.type is int:
                widget = QSpinBox()
                widget.setRange(-1_000_000, 1_000_000)
                widget.setValue(value)
            elif field.type is float:
                widget = QDoubleSpinBox()
                widget.setRange(-1_000_000.0, 1_000_000.0)
                widget.setDecimals(4)
                widget.setValue(value)
            elif isinstance(value, tuple) and field.name == "mutation":
                widget = QLineEdit(str(value))
                widget.setToolTip(self.tr("Enter as a tuple, e.g., (0.5, 1.0)"))
            elif field.type is bool:
                widget = QCheckBox()
                widget.setChecked(value)

            if widget:
                if isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                    widget.valueChanged.connect(self._update_input_summary_tab)
                elif isinstance(widget, QCheckBox):
                    widget.stateChanged.connect(self._update_input_summary_tab)
                elif isinstance(widget, QComboBox):
                    widget.currentTextChanged.connect(self._update_input_summary_tab)
                elif isinstance(widget, QLineEdit):
                    widget.textChanged.connect(self._update_input_summary_tab)

                input_dict[field.name] = widget
                form_layout.addRow(label_widget, widget)
                self.linked_min_max_widgets[form_name][field.name] = widget

        for min_key, max_key, relation in min_max_pairs:
            if (
                min_key in self.linked_min_max_widgets[form_name]
                and max_key in self.linked_min_max_widgets[form_name]
            ):
                min_widget = self.linked_min_max_widgets[form_name][min_key]
                max_widget = self.linked_min_max_widgets[form_name][max_key]
                if isinstance(min_widget, (QSpinBox, QDoubleSpinBox)) and isinstance(
                    max_widget, (QSpinBox, QDoubleSpinBox)
                ):
                    min_widget.valueChanged.connect(
                        partial(
                            self._update_min_max_validator, min_widget, max_widget, "min", relation
                        )
                    )
                    max_widget.valueChanged.connect(
                        partial(
                            self._update_min_max_validator, max_widget, min_widget, "max", relation
                        )
                    )
                    self._update_min_max_validator(
                        min_widget, max_widget, "min", relation, min_widget.value()
                    )

        self._update_input_summary_tab()

    def _update_min_max_validator(
        self,
        source_widget: QWidget,
        peer_widget: QWidget,
        source_type: str,
        relation: str,
        value: Any,
    ):
        """A generic validator to enforce min <= max constraints between two numeric QWidgets."""
        try:
            if source_type == "min":
                offset = 1 if relation == "<" else 0
                new_min = value + offset
                peer_widget.setMinimum(new_min)
            elif source_type == "max":
                offset = 1 if relation == "<" else 0
                new_max = value - offset
                peer_widget.setMaximum(new_max)
        except Exception:
            pass

    def _get_params_from_form(self, dc_class: type, input_dict: Dict) -> Optional[Any]:
        """Reads values from the UI widgets and creates a dataclass instance."""
        if not dc_class:
            return None
        try:
            kwargs = {}
            for name, widget in input_dict.items():
                if isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                    kwargs[name] = widget.value()
                elif isinstance(widget, QCheckBox):
                    kwargs[name] = widget.isChecked()
                elif isinstance(widget, QComboBox):
                    kwargs[name] = widget.currentText()
                elif isinstance(widget, QLineEdit) and name == "mutation":
                    try:
                        kwargs[name] = eval(widget.text())
                    except:
                        raise ValueError(
                            self.tr("Invalid format for mutation tuple. Use (min, max).")
                        )
            return dc_class(**kwargs)
        except Exception as e:
            QMessageBox.critical(
                self,
                self.tr("Settings Error"),
                self.tr("Invalid parameter input for {dc_class_name}:\n\n{error}").format(
                    dc_class_name=dc_class.__name__, error=e
                ),
            )
            logger.error(f"Error creating {dc_class.__name__} from form: {e}", exc_info=True)
            return None

    def _get_current_input_parameters(self) -> Dict[str, Dict[str, Any]]:
        """Gathers all current input parameters from the engine and UI forms for display or export."""
        if not self.engine:
            return {}
        all_params = {
            "General & Reservoir": {},
            "EOR Parameters": {},
            "Economic Parameters": {},
            "Genetic Algorithm": {},
            "Bayesian Optimization": {},
            "Particle Swarm Optimization": {},
            "Differential Evolution": {},
        }

        all_params["General & Reservoir"]["MMP (Calculated, psi)"] = self.engine.mmp
        all_params["General & Reservoir"]["Average Porosity"] = self.engine.avg_porosity
        all_params["General & Reservoir"]["OOIP (STB)"] = self.engine.reservoir.ooip_stb
        all_params["General & Reservoir"]["Project Lifetime (years)"] = (
            self.engine.operational_params.project_lifetime_years
        )

        for f in fields(self.engine.eor_params):
            all_params["EOR Parameters"][f.name] = getattr(self.engine.eor_params, f.name)
        for f in fields(self.engine.economic_params):
            all_params["Economic Parameters"][f.name] = getattr(self.engine.economic_params, f.name)

        def get_params_from_ui(tab_index, param_inputs, category_key):
            if self.hyperparam_tabs.isTabVisible(tab_index):
                for f_name, widget in param_inputs.items():
                    val = None
                    if isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                        val = widget.value()
                    elif isinstance(widget, QCheckBox):
                        val = widget.isChecked()
                    elif isinstance(widget, QComboBox):
                        val = widget.currentText()
                    elif isinstance(widget, QLineEdit):
                        val = widget.text()
                    if val is not None:
                        all_params[category_key][f_name] = val

        get_params_from_ui(0, self.ga_param_inputs, "Genetic Algorithm")
        get_params_from_ui(1, self.bo_param_inputs, "Bayesian Optimization")
        get_params_from_ui(2, self.pso_param_inputs, "Particle Swarm Optimization")
        get_params_from_ui(3, self.de_param_inputs, "Differential Evolution")

        return {k: v for k, v in all_params.items() if v}

    def _update_input_summary_tab(self):
        """Refreshes the tree view with the latest input parameter values."""
        self.input_params_tree.clear()
        if not self.engine:
            return

        param_data = self._get_current_input_parameters()
        for category, params in param_data.items():
            category_item = QTreeWidgetItem(self.input_params_tree)
            category_item.setText(0, self.tr(category))
            category_item.setExpanded(True)

            for key, value in sorted(params.items()):
                meta = self.PARAMETER_METADATA.get(key, {})
                display_name = self.tr(meta.get("display", key.replace("_", " ").title()))
                unit = meta.get("unit", "")
                value_str = f"{value:.4f}" if isinstance(value, float) else str(value)

                child_item = QTreeWidgetItem(category_item)
                child_item.setText(0, f"{display_name} {unit}".strip())
                child_item.setText(1, value_str)
        self.input_params_tree.resizeColumnToContents(0)

    def _run_optimization(self):
        """Validates inputs, gathers parameters, and starts the OptimizationWorker thread."""
        if not self.engine:
            QMessageBox.critical(
                self, self.tr("Engine Error"), self.tr("Optimization Engine is not available.")
            )
            return
        if self.worker and self.worker.isRunning():
            reply = QMessageBox.question(
                self,
                self.tr("Busy"),
                self.tr(
                    "An optimization run is already in progress. Do you want to stop it and start a new one?"
                ),
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.Yes:
                self.worker.stop()
            else:
                return

        self._clear_results(clear_inputs=False)
        method_text = self.method_combo.currentText()
        method_name = self.method_combo.currentData()
        self.engine.chosen_objective = self.objective_combo.currentData()

        kwargs = {}
        method_key_lower = method_name.lower()
        if "genetic" in method_key_lower or "hybrid" in method_key_lower:
            params = self._get_params_from_form(GeneticAlgorithmParams, self.ga_param_inputs)
            kwargs["ga_params_override"] = params
        if "bayesian" in method_key_lower or "hybrid" in method_key_lower:
            params = self._get_params_from_form(BayesianOptimizationParams, self.bo_param_inputs)
            kwargs["bo_params_override"] = params
        if "pso" in method_key_lower:
            params = self._get_params_from_form(ParticleSwarmParams, self.pso_param_inputs)
            kwargs["pso_params_override"] = params
        if "de" in method_key_lower:
            params = self._get_params_from_form(DifferentialEvolutionParams, self.de_param_inputs)
            kwargs["de_params_override"] = params

        if any(v is None for v in kwargs.values()):
            return

        self.results_tabs.setCurrentIndex(0)
        self.run_button.setEnabled(False)
        self.status_label.setText(
            self.tr("<i>Starting optimization with {method_text}...</i>").format(
                method_text=method_text
            )
        )

        self.worker = OptimizationWorker(self.engine, method_name, kwargs)
        self.worker.result_ready.connect(self._on_result)
        self.worker.error_occurred.connect(self._on_error)
        self.worker.finished.connect(self._on_worker_finished)
        self.worker.progress_updated.connect(self._on_progress_update)
        self.worker.ga_progress_updated.connect(self._update_convergence_live_plot)
        self.worker.target_unreachable.connect(self._handle_target_unreachable)
        self.worker.start()

    def _clear_results(self, clear_inputs=True):
        """Resets the results and plotting panels to their initial state."""
        self.summary_label.setText(self.tr("<i>Run optimization to view results.</i>"))
        self.results_table.setRowCount(0)
        if hasattr(self, "detailed_summary_table"):
            self.detailed_summary_table.setRowCount(0)
        if clear_inputs:
            self.input_params_tree.clear()
        self.analysis_plot_view.setHtml("")
        self.log_display.clear()
        self.results_tabs.setTabEnabled(4, False)  # Disable Analysis tab
        self.export_button.setEnabled(False)
        self.current_results = None
        self.convergence_live_data.clear()

    def _on_progress_update(self, message: str):
        """Updates the status label with progress messages from the worker."""
        self.status_label.setText(f"<i>{message}</i>")

    def _on_result(self, results: Dict[str, Any]):
        """Handles the successful completion of an optimization run."""
        self.current_results = results
        self.export_button.setEnabled(True)

        # --- NEW: Automatically run material balance analysis ---
        if self.engine:
            try:
                resolution = self.engine.operational_params.time_resolution
                mb_results = create_material_balance_from_optimization(
                    self.current_results,
                    resolution,
                    self.engine.eor_params.co2_density_tonne_per_mscf,
                    self.engine.co2_storage_params.leakage_rate_fraction,
                )
                if mb_results:
                    self.current_results["material_balance_analysis"] = mb_results
                    logger.info(
                        "Successfully generated material balance analysis post-optimization."
                    )
            except Exception as e:
                logger.error(f"Failed to generate material balance analysis: {e}", exc_info=True)
                self.current_results["material_balance_analysis"] = None
        # --- END NEW ---

        self._display_summary(results)
        self._display_dynamic_table(results)
        self._display_detailed_summary(results)
        self.optimization_completed.emit(results)
        self._setup_analysis_options()

        self.results_tabs.setCurrentIndex(2)  # Switch to Optimization Summary

        for i in range(self.plot_list.count()):
            item = self.plot_list.item(i)
            if item.text() == self.tr("Convergence"):
                self.plot_list.setCurrentItem(item)
                break

    def _display_dynamic_table(self, results: Dict[str, Any]):
        """Populates the results table with the optimized parameters."""
        params_to_show = results.get("optimized_params_final_clipped", {}).copy()
        self.results_table.setRowCount(0)
        self.results_table.setRowCount(len(params_to_show))
        sorted_params = sorted(params_to_show.items())

        for i, (key, val) in enumerate(sorted_params):
            meta = self.PARAMETER_METADATA.get(key, {})
            display_name = self.tr(meta.get("display", key.replace("_", " ").title()))
            unit = meta.get("unit", "")
            description = self.tr("Operational parameter optimized within standard bounds.")

            if self.engine and key in self.engine.RELAXABLE_CONSTRAINTS:
                description = self.tr(
                    self.engine.RELAXABLE_CONSTRAINTS[key].get(
                        "description", self.tr("Underlying parameter.")
                    )
                )
                if key in results.get("unlocked_params_in_run", []):
                    description += self.tr(
                        " (This parameter was unlocked and optimized to meet the target.)"
                    )

            val_str = f"{val:.4f}" if isinstance(val, (float, np.floating)) else str(val)
            self.results_table.setItem(i, 0, QTableWidgetItem(f"{display_name} {unit}".strip()))
            self.results_table.setItem(i, 1, QTableWidgetItem(val_str))
            self.results_table.setItem(i, 2, QTableWidgetItem(description))

        self.results_table.resizeRowsToContents()

    def _display_summary(self, results: Dict[str, Any]):
        """Displays a high-level summary of the optimization results."""
        final_metrics = results.get("final_metrics", {})
        obj_name = (
            self.engine.chosen_objective.replace("_", " ").title() if self.engine else "Objective"
        )
        obj_val = final_metrics.get(self.engine.chosen_objective, "N/A")

        rf = final_metrics.get("recovery_factor", "N/A")
        npv = final_metrics.get("npv", "N/A")
        co2_util = final_metrics.get("co2_utilization", "N/A")
        method = results.get("method", "N/A").replace("_", " ").title()

        obj_val_str = (
            f"{obj_val:.4g}" if isinstance(obj_val, (float, np.floating)) else str(obj_val)
        )
        rf_str = f"{rf:.4f}" if isinstance(rf, (float, np.floating)) else str(rf)
        npv_str = f"${npv:,.0f}" if isinstance(npv, (float, np.floating)) else str(npv)
        co2_util_str = (
            f"{co2_util:.2f} MSCF/STB"
            if isinstance(co2_util, (float, np.floating))
            else str(co2_util)
        )

        summary = (
            f"<b>{self.tr('Optimization Complete')}: {self.tr(method)}</b><br>"
            f"{self.tr('Final Optimized')} {self.tr(obj_name)}: <b>{obj_val_str}</b><br><br>"
            f"<b>{self.tr('Key Performance Indicators')}:</b><br>"
            f"&nbsp;&nbsp;&nbsp;&nbsp;{self.tr('Net Present Value (NPV)')}: <b>{npv_str}</b><br>"
            f"&nbsp;&nbsp;&nbsp;&nbsp;{self.tr('Recovery Factor')}: <b>{rf_str}</b><br>"
            f"&nbsp;&nbsp;&nbsp;&nbsp;{self.tr('CO₂ Utilization')}: <b>{co2_util_str}</b>"
        )
        self.summary_label.setText(summary)

    def _display_detailed_summary(self, results: Dict[str, Any]):
        """Populates the new table with year-by-year profile data for scientific use."""
        profiles = results.get("optimized_profiles")
        if not profiles:
            self.detailed_summary_table.clear()
            self.detailed_summary_table.setRowCount(1)
            self.detailed_summary_table.setColumnCount(1)
            self.detailed_summary_table.setItem(
                0, 0, QTableWidgetItem(self.tr("No detailed profile data available."))
            )
            return

        try:
            resolution = self.engine.operational_params.time_resolution if self.engine else "yearly"
            # Filter to only profiles for the selected resolution
            res_profiles = {k: v for k, v in profiles.items() if k.startswith(resolution)}
            df = pd.DataFrame(res_profiles)
            df.index = df.index + 1
            df.index.name = self.tr(resolution.title())

            # Rename columns to be more readable
            new_columns = []
            for col in df.columns:
                base_key = col.replace(f"{resolution}_", "")
                meta = self.PARAMETER_METADATA.get(base_key, {})
                display_name = self.tr(meta.get("display", base_key.replace("_", " ").title()))
                unit = meta.get("unit", "")
                new_columns.append(f"{display_name} {unit}".strip())
            df.columns = new_columns

            self.detailed_summary_table.setRowCount(df.shape[0])
            self.detailed_summary_table.setColumnCount(df.shape[1])
            self.detailed_summary_table.setHorizontalHeaderLabels(list(df.columns))

            for row_idx, row in enumerate(df.itertuples()):
                for col_idx, value in enumerate(row[1:]):
                    if isinstance(value, (float, np.floating)):
                        item = QTableWidgetItem(f"{value:,.2f}")
                    else:
                        item = QTableWidgetItem(str(value))
                    self.detailed_summary_table.setItem(row_idx, col_idx, item)

            self.detailed_summary_table.resizeColumnsToContents()
        except Exception as e:
            logger.error(f"Failed to display detailed summary: {e}", exc_info=True)
            self.detailed_summary_table.clear()
            self.detailed_summary_table.setRowCount(1)
            self.detailed_summary_table.setColumnCount(1)
            self.detailed_summary_table.setItem(
                0, 0, QTableWidgetItem(self.tr("Error displaying profile data: {e}").format(e=e))
            )

    def _get_graphing_config(self) -> Dict[str, Any]:
        if hasattr(self.parent(), "graphing_config"):
            return self.parent().graphing_config
        return {}

    @pyqtSlot(dict)
    def _update_convergence_live_plot(self, progress_data: Dict[str, Any]):
        """Updates the live convergence plot during a run."""
        self.convergence_live_data.append(progress_data)
        iterations = [d.get("iteration", d.get("generation")) for d in self.convergence_live_data]
        best_fitness = [d["best_fitness"] for d in self.convergence_live_data]

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=iterations, y=best_fitness, mode="lines+markers", name=self.tr("Best Fitness")
            )
        )

        fig.update_layout(
            title_text=self.tr("Live Optimization Progress"),
            xaxis_title_text=self.tr("Function Evaluations"),
            yaxis_title_text=self.tr("Objective Value"),
            legend_title_text=self.tr("Metric"),
        )
        self.analysis_plot_view.setHtml(
            fig.to_html(include_plotlyjs="cdn", config=self._get_graphing_config())
        )

    def _setup_analysis_options(self):
        """Populates the plot selection list after a run is complete."""
        self.results_tabs.setTabEnabled(4, True)  # Enable Optimization Analysis tab
        self.plot_list.clear()

        plot_options = ["Convergence", "Final Production Profiles", "Parameter Sensitivity"]
        if self.current_results and "bayes_opt_obj" in self.current_results:
            plot_options.append("Objective vs. Parameter (BO)")

        # --- NEW: Add CO2 and Material Balance analysis options ---
        if self.current_results and self.current_results.get("material_balance_analysis"):
            plot_options.extend(
                [
                    "CO2 Performance Summary",
                    "Material Balance",
                    "Storage Efficiency",
                    "Cumulative CO2 Balance",
                ]
            )

        if self.current_results and self.current_results.get("dca_results"):
            plot_options.append("Decline Curve Analysis")
        if self.current_results and self.current_results.get("method") == "genetic_algorithm":
            plot_options.append("GA Coverage Distribution")
        # --- END NEW ---

        for option in plot_options:
            item = QListWidgetItem(QIcon.fromTheme("view-plot"), self.tr(option))
            self.plot_list.addItem(item)

        self._populate_sensitivity_combo()
        self._populate_bo_sensitivity_combo()

    def _on_analysis_plot_selected(
        self, current_item: QListWidgetItem, previous_item: QListWidgetItem
    ):
        """Handles the logic for when a new plot type is selected from the list."""
        if not current_item:
            self.plot_controls_widget.setVisible(False)
            return

        plot_type = current_item.text()
        show_sensitivity = plot_type == self.tr("Parameter Sensitivity")
        show_bo_sensitivity = plot_type == self.tr("Objective vs. Parameter (BO)")

        self.sensitivity_param_combo.setVisible(show_sensitivity)
        self.bo_sensitivity_param_combo.setVisible(show_bo_sensitivity)
        self.plot_controls_widget.setVisible(show_sensitivity or show_bo_sensitivity)

        self._generate_selected_plot()

    def _populate_sensitivity_combo(self):
        """Populates the dropdown with parameters available for sensitivity analysis."""
        self.sensitivity_param_combo.clear()
        if not self.current_results:
            return

        params = list(self.current_results.get("optimized_params_final_clipped", {}).keys())
        econ_params = [f.name for f in fields(EconomicParameters)]
        all_param_keys = sorted(list(set(params + econ_params)))

        for key in all_param_keys:
            meta = self.PARAMETER_METADATA.get(key, {})
            display_name = self.tr(meta.get("display", key.replace("_", " ").title()))
            self.sensitivity_param_combo.addItem(display_name, userData=key)

    def _populate_bo_sensitivity_combo(self):
        """Populates the dropdown for the Bayesian Optimization results plot."""
        self.bo_sensitivity_param_combo.clear()
        if not (self.current_results and "bayes_opt_obj" in self.current_results):
            return
        bo = self.current_results["bayes_opt_obj"]
        if not (hasattr(bo, "res") and bo.res and "params" in bo.res[0]):
            return

        params = list(bo.res[0]["params"].keys())
        for key in sorted(params):
            meta = self.PARAMETER_METADATA.get(key, {})
            display_name = self.tr(meta.get("display", key.replace("_", " ").title()))
            self.bo_sensitivity_param_combo.addItem(display_name, userData=key)

    def _generate_co2_summary_html(self) -> str:
        """Generates an HTML summary of CO2 performance metrics."""
        if not self.current_results or not self.current_results.get("material_balance_analysis"):
            return f"<p>{self.tr('No CO2 analysis data available.')}</p>"

        mb_stats = self.current_results["material_balance_analysis"].get("summary_statistics", {})
        final_metrics = self.current_results.get("final_metrics", {})

        def get_stat(key, unit, is_money=False):
            val = mb_stats.get(key)
            if val is None:
                return "N/A"
            prefix = "$" if is_money else ""
            return f"{prefix}{val:,.0f} {unit}"

        co2_util = final_metrics.get("co2_utilization", "N/A")
        co2_util_str = (
            f"{co2_util:.2f} MSCF/STB" if isinstance(co2_util, (float, np.floating)) else "N/A"
        )

        total_purchased_val = mb_stats.get("total_injected_tonne", 0) - mb_stats.get(
            "total_recycled_tonne", 0
        )
        total_purchased_str = f"{total_purchased_val:,.2f} tonnes"

        html = """
        <style>
            body {{ font-family: sans-serif; margin: 20px; }}
            h2 {{ color: #333; }}
            table {{ border-collapse: collapse; width: 80%; margin-top: 20px; }}
            th, td {{ border: 1px solid #ddd; padding: 10px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
        </style>
        <h2>CO₂ Performance Summary</h2>
        <table>
            <tr><th>Metric</th><th>Value</th></tr>
            <tr><td>CO₂ Utilization Factor</td><td><b>{co2_util_str}</b></td></tr>
            <tr><td>Total CO₂ Injected</td><td>{total_injected}</td></tr>
            <tr><td>Total CO₂ Produced</td><td>{total_produced}</td></tr>
            <tr><td>Total CO₂ Recycled</td><td>{total_recycled}</td></tr>
            <tr><td>Total CO₂ Purchased</td><td>{total_purchased}</td></tr>
            <tr><td><b>Total Net CO₂ Stored</b></td><td><b>{total_stored}</b></td></tr>
            <tr><td>Total CO₂ Leaked</td><td>{total_leaked}</td></tr>
            <tr><td>Final Cumulative Stored</td><td>{final_cumulative}</td></tr>
            <tr><td>Average Storage Efficiency</td><td>{avg_efficiency:.2f}%</td></tr>
        </table>
        """.format(
            co2_util_str=co2_util_str,
            total_injected=get_stat("total_injected_tonne", "tonnes"),
            total_produced=get_stat("total_produced_tonne", "tonnes"),
            total_recycled=get_stat("total_recycled_tonne", "tonnes"),
            total_purchased=total_purchased_str,
            total_stored=get_stat("total_net_stored_tonne", "tonnes"),
            total_leaked=get_stat("total_leakage_tonne", "tonnes"),
            final_cumulative=get_stat("final_cumulative_stored_tonne", "tonnes"),
            avg_efficiency=mb_stats.get("avg_storage_efficiency", 0) * 100,
        )
        return html

    def _generate_selected_plot(self):
        """Generates and displays the plot currently selected in the QListWidget."""
        if not self.engine or not self.current_results or not self.plot_list.currentItem():
            return

        plot_type = self.plot_list.currentItem().text()
        fig = None

        try:
            mb_analysis = self.current_results.get("material_balance_analysis")

            if plot_type == self.tr("Convergence"):
                fig = self.engine.plotting_manager.plot_optimization_convergence(self.current_results)
            elif plot_type == self.tr("Parameter Sensitivity"):
                param_key = self.sensitivity_param_combo.currentData()
                if param_key:
                    fig = self.engine.plotting_manager.plot_parameter_sensitivity(param_key, self.current_results)
            elif plot_type == self.tr("Final Production Profiles"):
                fig = self.engine.plotting_manager.plot_production_profiles(self.current_results)
                if fig:
                    resolution = self.engine.operational_params.time_resolution
                    fig.update_layout(
                        title_text=self.tr("Optimized {resolution} Profiles").format(
                            resolution=resolution.title()
                        ),
                        xaxis_title=self.tr("Project {resolution}").format(
                            resolution=resolution.title()
                        ),
                    )
            elif plot_type == self.tr("Objective vs. Parameter (BO)"):
                param_key = self.bo_sensitivity_param_combo.currentData()
                if param_key:
                    fig = self.engine.plotting_manager.plot_objective_vs_parameter(param_key, self.current_results)
            elif plot_type == self.tr("CO2 Performance Summary"):
                fig = self.engine.plotting_manager.plot_co2_performance_summary_table(self.current_results)
            elif plot_type == self.tr("Material Balance") and mb_analysis:
                fig = mb_analysis["graphs"]["main_balance"]
            elif plot_type == self.tr("Storage Efficiency") and mb_analysis:
                fig = mb_analysis["graphs"]["efficiency"]
            elif plot_type == self.tr("Cumulative CO2 Balance") and mb_analysis:
                fig = mb_analysis["graphs"]["cumulative"]
            elif plot_type == self.tr("Decline Curve Analysis") and self.current_results.get(
                "dca_results"
            ):
                fig = self.engine.dca_analyzer.plot_decline_curve(
                    self.current_results["dca_results"]
                )

            if fig:
                self.analysis_plot_view.setHtml(
                    fig.to_html(include_plotlyjs="cdn", config=self._get_graphing_config())
                )
            else:
                if plot_type in [
                    self.tr("Parameter Sensitivity"),
                    self.tr("Objective vs. Parameter (BO)"),
                ]:
                    self.analysis_plot_view.setHtml(
                        f"<p style='font-family:sans-serif; text-align:center; margin-top:50px;'>{self.tr('Select a parameter to generate the plot.')}</p>"
                    )
                else:
                    self.analysis_plot_view.setHtml("")
        except Exception as e:
            logger.error(f"Error generating analysis plot '{plot_type}': {e}", exc_info=True)
            QMessageBox.critical(
                self,
                self.tr("Plotting Error"),
                self.tr("An error occurred while generating the plot: {e}").format(e=e),
            )
            self.analysis_plot_view.setHtml(
                f"<p style='color:red; font-family:sans-serif; text-align:center; margin-top:50px;'>{self.tr('Could not generate plot for')} '{plot_type}'.<br>Error: {e}</p>"
            )

    @pyqtSlot(dict)
    def _handle_target_unreachable(self, failed_results: dict):
        """Handles the case where the optimizer could not meet a specified target."""
        if not self.engine:
            return

        self.status_label.setText(
            self.tr("<b style='color:orange;'>Target was not reached with current constraints.</b>")
        )
        self._on_result(failed_results)

        target_name = failed_results.get("target_objective_name_in_run", "target").replace("_", " ")
        achieved_val = failed_results.get("final_target_value_achieved", 0.0)
        target_val = failed_results.get("target_objective_value_in_run", 0.0)

        reply = QMessageBox.information(
            self,
            self.tr("Target Unreachable"),
            self.tr(
                "The optimizer could not meet the specified {target_name} target of {target_val:.4f}.\n\n"
                "The closest value found was {achieved_val:.4f}.\n\n"
                "Would you like to unlock additional reservoir or EOR parameters to allow the optimizer more freedom to meet the target?"
            ).format(target_name=target_name, target_val=target_val, achieved_val=achieved_val),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )

        if reply == QMessageBox.StandardButton.Yes:
            available_params = list(self.engine.RELAXABLE_CONSTRAINTS.keys())
            dialog = UnlockParametersDialog(available_params, self)
            if dialog.exec():
                unlocked_params = dialog.get_selected_parameters()
                if unlocked_params:
                    logger.info(f"User chose to unlock parameters for re-run: {unlocked_params}")
                    self._rerun_optimization_with_unlocked_params(unlocked_params)
                else:
                    self.status_label.setText(
                        self.tr("<i>Re-run cancelled. No parameters were unlocked.</i>")
                    )

    def _rerun_optimization_with_unlocked_params(self, unlocked_params: List[str]):
        """Initiates a new optimization run with certain fixed parameters now variable."""
        if not self.engine:
            return

        logger.info(f"Preparing for re-optimization with unlocked parameters: {unlocked_params}")
        self.engine.prepare_for_rerun_with_unlocked_params(unlocked_params)
        self._clear_results(clear_inputs=False)

        self._run_optimization()

    def _on_error(self, error_msg: str):
        """Handles errors emitted from the worker thread."""
        self._clear_results()
        self.status_label.setText(
            f"<p style='color:red;'><b>{self.tr('Error')}:</b> {error_msg}</p>"
        )
        self.run_button.setEnabled(self.engine is not None)

    def _on_worker_finished(self):
        """Cleans up after the worker thread has finished."""
        self.run_button.setEnabled(self.engine is not None)
        if self.engine:
            self.engine.reset_to_base_state()
            logger.info("Engine state reset to base configuration after worker finished.")

        if self.worker and not self.worker.was_successful():
            if self.tr("Error") not in self.status_label.text():
                self.status_label.setText(
                    self.tr("<i>Optimization stopped or finished with issues. Check log.</i>")
                )
        elif self.current_results:
            if self.current_results.get("target_was_unreachable"):
                self.status_label.setText(
                    self.tr(
                        "<i><b style='color:orange;'>Re-optimization finished. Target may still be unreachable.</b></i>"
                    )
                )
            else:
                self.status_label.setText(self.tr("<i>Optimization finished successfully.</i>"))
        else:
            if self.tr("Error") not in self.status_label.text():
                self.status_label.setText(
                    self.tr("<i>Optimization worker finished. Ready for next run.</i>")
                )

        if self.worker:
            self.worker.deleteLater()
            self.worker = None

    def on_engine_type_changed(self, engine_type: str):
        """Called when engine type changes from Config Widget (single source of truth).

        Note: The engine selection widget has been moved to Config Widget.
        This method is called by MainWindow when engine type changes.

        Args:
            engine_type: The new engine type ("simple", "detailed", or "surrogate")
        """
        try:
            logger.info(f"OptimizationWidget: Engine type changed to '{engine_type}'")

            # Store the selected engine type
            self.selected_engine_type = engine_type

            # Update status
            self.status_label.setText(
                f"<i>Engine changed to {engine_type} (from Config Widget). Use Apply to update optimization engine.</i>"
            )

        except Exception as e:
            logger.error(f"Error handling engine type change: {e}", exc_info=True)
            logger.info(f"Engine type change requested: {engine_type} (use_simple_physics={use_simple})")

            QMessageBox.information(
                self,
                "Engine Switching",
                f"Switching to {engine_type} engine.\n\n"
                "The optimization engine will be recreated with the new engine type.",
            )

        except Exception as e:
            error_msg = f"Error selecting engine: {str(e)}"
            QMessageBox.critical(self, "Engine Selection Error", error_msg)
            self.status_label.setText(f"<i><b style='color:red;'>{error_msg}</b></i>")
            logger.error(error_msg, exc_info=True)

    def _clean_dict_for_json(self, data_dict: Any) -> Any:
        """Recursively cleans a dictionary to make it JSON serializable."""
        if isinstance(data_dict, dict):
            cleaned_dict = {}
            for key, value in data_dict.items():
                if key in ["bayes_opt_obj", "pygad_instance", "de_result_obj", "charts"]:
                    cleaned_dict[key] = f"<{type(value).__name__} object not serialized>"
                    continue
                cleaned_dict[key] = self._clean_dict_for_json(value)
            return cleaned_dict
        elif isinstance(data_dict, list):
            return [self._clean_dict_for_json(item) for item in data_dict]
        elif hasattr(data_dict, "tolist"):
            return data_dict.tolist()
        elif isinstance(data_dict, (np.integer, np.floating)):
            return data_dict.item()
        elif is_dataclass(data_dict):
            return asdict(data_dict)
        else:
            return data_dict

    def _export_run_data(self):
        """Exports the current run's inputs, results, and graphs to a new folder."""
        if not self.current_results:
            QMessageBox.warning(
                self,
                self.tr("No Data"),
                self.tr("There is no result data to export. Please run an optimization first."),
            )
            return

        start_dir = str(Path.home())
        target_dir = QFileDialog.getExistingDirectory(
            self, self.tr("Select Directory to Save Export Folder"), start_dir
        )
        if not target_dir:
            return

        try:
            method = self.current_results.get("method", "optimization").replace("_", "-")
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            export_folder_name = f"Export-{method}-{timestamp}"
            export_path = Path(target_dir) / export_folder_name
            export_path.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            QMessageBox.critical(
                self,
                self.tr("Export Error"),
                self.tr("Could not create export directory.\nError: {e}").format(e=e),
            )
            return

        # --- Save Comprehensive JSON Data ---
        try:
            json_data = {
                "report_metadata": {
                    "generation_timestamp": datetime.now().isoformat(),
                    "method": self.current_results.get("method"),
                    "objective": self.objective_combo.currentText(),
                },
                "input_parameters": self._get_current_input_parameters(),
                "full_results": self._clean_dict_for_json(self.current_results),
            }
            json_file_path = export_path / "full_run_data.json"
            with open(json_file_path, "w", encoding="utf-8") as f:
                json.dump(json_data, f, indent=4)
        except Exception as e:
            logger.error(f"Failed to write full JSON data: {e}", exc_info=True)

        # --- Save Detailed CSV Summaries ---
        try:
            dataframes = self._generate_detailed_csv_export()
            for name, df in dataframes.items():
                csv_file_path = export_path / f"summary_{name}.csv"
                df.to_csv(csv_file_path)
        except Exception as e:
            logger.error(f"Failed to write detailed CSV summaries: {e}", exc_info=True)

        # --- Save Text Summary ---
        try:
            summary_content = self._generate_detailed_txt_export()
            summary_file_path = export_path / "results_summary.txt"
            with open(summary_file_path, "w", encoding="utf-8") as f:
                f.write(summary_content)
        except Exception as e:
            logger.error(f"Failed to write summary text file: {e}", exc_info=True)

        # --- Generate and Save Graphs ---
        if self.engine:
            plots_to_generate = self._get_all_available_plots()
            for plot_name, plot_info in plots_to_generate.items():
                try:
                    fig = plot_info["func"](self.current_results)
                    if fig:
                        image_path = export_path / f"{plot_name}.png"
                        width = plot_info.get("width", 1200)
                        height = plot_info.get("height", 800)
                        fig.write_image(str(image_path), width=width, height=height)
                except Exception as e:
                    logger.error(
                        f"Failed to generate or save plot '{plot_name}': {e}", exc_info=True
                    )

        self.status_label.setText(
            self.tr("<i>Run data successfully exported to {export_path}.</i>").format(
                export_path=export_path
            )
        )
        logger.info(f"Run data exported to {export_path}")

    def _get_all_available_plots(self) -> Dict[str, Dict]:
        """Returns a dictionary of all possible plots and their generation functions for export."""
        if not self.engine or not self.current_results:
            return {}

        plots = {
            "convergence": {"func": self.engine.plotting_manager.plot_optimization_convergence},
            "production_profiles": {"func": self.engine.plotting_manager.plot_production_profiles},
            "co2_summary_table": {
                "func": self.engine.plotting_manager.plot_co2_performance_summary_table,
                "width": 800,
                "height": 400,
            },
        }

        mb_analysis = self.current_results.get("material_balance_analysis")
        if mb_analysis and "graphs" in mb_analysis:
            plots["material_balance"] = {
                "func": lambda r: r["material_balance_analysis"]["graphs"]["main_balance"]
            }
            plots["storage_efficiency"] = {
                "func": lambda r: r["material_balance_analysis"]["graphs"]["efficiency"]
            }
            plots["cumulative_co2_balance"] = {
                "func": lambda r: r["material_balance_analysis"]["graphs"]["cumulative"]
            }

        if self.current_results.get("dca_results"):
            plots["decline_curve_analysis"] = {
                "func": lambda r: self.engine.dca_analyzer.plot_decline_curve(r["dca_results"])
            }

        return plots

    def _generate_detailed_csv_export(self) -> Dict[str, pd.DataFrame]:
        """Generates detailed CSV exports of the optimization results for both yearly and daily resolutions."""
        if not (self.current_results and self.engine):
            return {}

        profiles = self.current_results.get("optimized_profiles", {})
        mb_data = self.current_results.get("material_balance_analysis", {}).get(
            "material_balance_data", {}
        )

        dataframes = {}
        for resolution in ["yearly", "daily"]:
            res_profiles = {k: v for k, v in profiles.items() if k.startswith(resolution)}
            if not res_profiles:
                continue

            df = pd.DataFrame(res_profiles)
            df.index = df.index + 1
            df.index.name = resolution.title()

            # Merge material balance data if available
            if mb_data and resolution == self.engine.operational_params.time_resolution:
                mb_df = pd.DataFrame(mb_data)
                mb_df.index = mb_df.index + 1
                mb_df = mb_df.drop(columns=["years"], errors="ignore")
                df = df.join(mb_df)

            # Clean up column names
            df.columns = [
                col.replace(f"{resolution}_", "").replace("_", " ").title() for col in df.columns
            ]
            dataframes[resolution] = df

        return dataframes

    def _generate_detailed_txt_export(self) -> str:
        """Generates a detailed TXT export with enhanced statistics and input parameters."""
        output = io.StringIO()

        # --- HEADER ---
        output.write(f"=== CO₂ EOR OPTIMIZATION RUN REPORT ===\n")
        output.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        output.write(
            f"Method: {self.current_results.get('method', 'Unknown').replace('_', ' ').title()}\n"
        )
        output.write(f"Objective: {self.objective_combo.currentText()}\n\n")

        # --- INPUTS ---
        output.write("=== INPUT PARAMETERS ===\n")
        input_params = self._get_current_input_parameters()
        user_defined_params = self._get_user_defined_params()

        for category, params in input_params.items():
            output.write(f"\n--- {category.upper()} ---\n")
            for param_name, param_value in sorted(params.items()):
                flag = " [USER-DEFINED]" if param_name in user_defined_params else ""
                output.write(f"{param_name}: {param_value}{flag}\n")

        # --- RESULTS ---
        output.write(f"\n\n=== OPTIMIZATION RESULTS ===\n")
        output.write(
            f"Final Optimized {self.objective_combo.currentText()}: {self.current_results.get('objective_function_value', 'N/A'):.4g}\n"
        )

        output.write(f"\n--- ALL FINAL METRICS ---\n")
        final_metrics = self.current_results.get("final_metrics", {})
        for key, value in sorted(final_metrics.items()):
            display_name = key.replace("_", " ").title()
            value_str = f"{value:.4f}" if isinstance(value, (float, np.floating)) else str(value)
            output.write(f"{display_name}: {value_str}\n")

        output.write(f"\n--- OPTIMIZED PARAMETERS ---\n")
        for i in range(self.results_table.rowCount()):
            param = self.results_table.item(i, 0).text()
            value = self.results_table.item(i, 1).text()
            output.write(f"{param}: {value}\n")

        # --- ANALYSIS SUMMARIES ---
        if self.current_results.get("material_balance_analysis"):
            output.write(f"\n--- MATERIAL BALANCE SUMMARY ---\n")
            mb_stats = self.current_results["material_balance_analysis"].get(
                "summary_statistics", {}
            )
            for key, value in sorted(mb_stats.items()):
                output.write(f"{key.replace('_', ' ').title()}: {value}\n")

        if self.current_results.get("dca_results"):
            output.write(f"\n--- DECLINE CURVE ANALYSIS SUMMARY ---\n")
            dca_stats = self.current_results["dca_results"].get("summary", {})
            for key, value in sorted(dca_stats.items()):
                output.write(f"{key.replace('_', ' ').title()}: {value}\n")

        # --- OPTIMIZER STATS ---
        output.write(f"\n\n=== DETAILED OPTIMIZATION STATISTICS ===\n")
        stats = self.current_results.get("bo_statistics") or self.current_results.get(
            "ga_statistics"
        )
        if stats:
            for key, value in sorted(stats.items()):
                if isinstance(value, list):
                    continue
                output.write(f"{key.replace('_', ' ').title()}: {value}\n")

        # --- FOOTER ---
        output.write(f"\n\n=== END OF REPORT ===\n")
        output.write("Generated by CO₂ EOR Optimization Tool\n")

        return output.getvalue()

    def _get_user_defined_params(self) -> set:
        """Compares current engine parameters to base project parameters to find user customizations."""
        if not self.engine:
            return set()

        user_defined = set()

        # Define which dataclasses to check
        param_sets = [
            ("EOR Parameters", self.engine.eor_params, self.engine._base_eor_params),
            ("Economic Parameters", self.engine.economic_params, self.engine._base_economic_params),
        ]

        for category, current_params, base_params in param_sets:
            if not is_dataclass(current_params) or not is_dataclass(base_params):
                continue

            current_dict = asdict(current_params)
            base_dict = asdict(base_params)

            for key, current_val in current_dict.items():
                base_val = base_dict.get(key)
                # Check for significant difference, especially for floats
                if isinstance(current_val, float):
                    if not np.isclose(current_val, base_val):
                        user_defined.add(key)
                elif current_val != base_val:
                    user_defined.add(key)

        return user_defined

    # --- Methods for Integrated MMP Analysis ---
    def _mmp_normalize_gas_composition(self):
        co2 = self.mmp_co2_comp_input.value()
        ch4 = self.mmp_ch4_comp_input.value()
        n2 = self.mmp_n2_comp_input.value()
        total = co2 + ch4 + n2
        if total > 1e-6:
            self.mmp_co2_comp_input.setValue(co2 / total)
            self.mmp_ch4_comp_input.setValue(ch4 / total)
            self.mmp_n2_comp_input.setValue(n2 / total)

    def _mmp_update_method_specific_inputs_visibility(self):
        selected_method = self.mmp_method_combo.currentData()
        show_c7_mw = selected_method in ["hybrid_gh", "alston"]
        show_gas_comp = selected_method in ["yuan", "alston"]

        self.mmp_c7_mw_widget.setVisible(show_c7_mw)
        self.mmp_gas_comp_widget.setVisible(show_gas_comp)
        self.mmp_method_inputs_group.setVisible(show_c7_mw or show_gas_comp)

    def _mmp_update_method_availability(self):
        current_well = self.mmp_well_select_combo.currentData()
        if not isinstance(current_well, WellData):
            for i in range(self.mmp_method_combo.count()):
                if self.mmp_method_combo.itemData(i) not in ["auto", "cronquist"]:
                    self.mmp_method_combo.model().item(i).setEnabled(False)
            return

        for i in range(self.mmp_method_combo.count()):
            self.mmp_method_combo.model().item(i).setEnabled(True)
        self._mmp_update_method_specific_inputs_visibility()

    def _mmp_update_button_state(self):
        can_calculate = bool(self.well_data_list and self.pvt_data)
        self.mmp_calculate_button.setEnabled(can_calculate)
        self.mmp_status_label.setText(
            self.tr("<i>Ready to calculate MMP profile.</i>")
            if can_calculate
            else self.tr("<i>Well and/or PVT data is missing.</i>")
        )

    def _run_mmp_calculation(self):
        if self.mmp_worker and self.mmp_worker.isRunning():
            QMessageBox.warning(
                self, self.tr("Busy"), self.tr("An MMP analysis is already in progress.")
            )
            return

        well = self.mmp_well_select_combo.currentData()
        method = self.mmp_method_combo.currentData()
        if not all([well, self.pvt_data, method]):
            QMessageBox.warning(
                self,
                self.tr("Missing Data"),
                self.tr("A valid well, PVT data, and calculation method are required."),
            )
            return

        worker_kwargs = {"method": method}
        if self.mmp_c7_mw_widget.isVisible():
            worker_kwargs["c7_plus_mw_override"] = self.mmp_c7_mw_input.value()
        if self.mmp_gas_comp_widget.isVisible():
            self._mmp_normalize_gas_composition()
            gas_comp = {
                "CO2": self.mmp_co2_comp_input.value(),
                "CH4": self.mmp_ch4_comp_input.value(),
                "N2": self.mmp_n2_comp_input.value(),
            }
            worker_kwargs["gas_composition"] = {k: v for k, v in gas_comp.items() if v > 1e-6}

        self.mmp_status_label.setText(
            self.tr("<i>Calculating MMP profile for {well_name}...</i>").format(well_name=well.name)
        )
        self.mmp_calculate_button.setEnabled(False)
        self.mmp_plot_view.setHtml(
            f"<p style='text-align:center;'><i>{self.tr('Processing...')}</i></p>"
        )
        self.results_tabs.setCurrentIndex(1)  # Switch to MMP results tab

        self.mmp_worker = WellAnalysisWorker(
            well_data=well, pvt_data=self.pvt_data, **worker_kwargs
        )
        self.mmp_worker.result_ready.connect(self._on_mmp_result)
        self.mmp_worker.error_occurred.connect(self._on_mmp_error)
        self.mmp_worker.finished.connect(self._on_mmp_worker_finished)
        self.mmp_worker.start()

    def _on_mmp_result(self, results: Dict[str, list]):
        well_name = self.mmp_well_select_combo.currentText()
        self.mmp_status_label.setText(
            self.tr("MMP profile calculation complete for {well_name}.").format(well_name=well_name)
        )
        try:
            processed_results = {
                key: np.array(value) for key, value in results.items() if isinstance(value, list)
            }

            num_points = processed_results.get("depths", np.array([])).size
            if num_points > 0:
                self.mmp_plot_view.setHtml(
                    self._create_mmp_plot(processed_results, well_name).to_html(
                        include_plotlyjs="cdn", config=self._get_graphing_config()
                    )
                )
                self._populate_mmp_results_table(processed_results)

                mmp_values = processed_results.get("mmp", np.array([]))
                if mmp_values.size > 0 and np.any(~np.isnan(mmp_values)):
                    representative_mmp = np.nanmean(mmp_values)
                    logger.info(f"Emitting representative MMP: {representative_mmp:.2f} psi")
                    self.representative_mmp_calculated.emit(representative_mmp)
            else:
                self._on_mmp_error(self.tr("Analysis returned no data points."))
        except Exception as e:
            logger.error(f"Error processing MMP results: {e}", exc_info=True)
            self._on_mmp_error(self.tr("Failed to process results: {e}").format(e=e))

    def _on_mmp_error(self, error_message: str):
        self.mmp_status_label.setText(
            f"<p style='color:red;'><b>{self.tr('Error')}:</b> {error_message}</p>"
        )
        self.mmp_plot_view.setHtml(
            f"<p style='text-align:center; color:red;'>{self.tr('Calculation failed')}.<br>{error_message}</p>"
        )
        logger.error(f"MMP calculation error: {error_message}")

    def _on_mmp_worker_finished(self):
        self._mmp_update_button_state()
        if self.mmp_worker:
            self.mmp_worker.deleteLater()
            self.mmp_worker = None

    def _populate_mmp_results_table(self, results: Dict[str, np.ndarray]):
        self.mmp_results_table.setRowCount(0)
        depths, mmp, temp, api = (
            results.get(k, np.array([])) for k in ["depths", "mmp", "temperature", "api"]
        )
        if depths.size == 0:
            return

        valid_mask = ~np.isnan(mmp)
        if not np.any(valid_mask):
            return

        unique_rows = []
        if valid_mask.any():
            last_props = ()
            for i in range(len(depths)):
                if valid_mask[i]:
                    current_props = (temp[i], api[i], mmp[i])
                    if not last_props or not np.allclose(current_props, last_props[1:]):
                        row_to_add = (depths[i], temp[i], api[i], mmp[i])
                        unique_rows.append(row_to_add)
                        last_props = row_to_add

        self.mmp_results_table.setRowCount(len(unique_rows))
        for row_idx, (d, t, a, m) in enumerate(unique_rows):
            self.mmp_results_table.setItem(row_idx, 0, QTableWidgetItem(f"{d:.1f}"))
            self.mmp_results_table.setItem(row_idx, 1, QTableWidgetItem(f"{t:.2f}"))
            self.mmp_results_table.setItem(row_idx, 2, QTableWidgetItem(f"{a:.2f}"))
            self.mmp_results_table.setItem(row_idx, 3, QTableWidgetItem(f"{m:.2f}"))
        self.mmp_results_table.resizeColumnsToContents()

    def _create_mmp_plot(self, results: Dict[str, np.ndarray], well_name: str) -> go.Figure:
        depths, mmp, temp, api = (results.get(k) for k in ["depths", "mmp", "temperature", "api"])
        if depths is None or depths.size == 0 or mmp is None or mmp.size == 0:
            raise ValueError("Result dictionary is missing 'depths' or 'mmp' data.")

        fig = go.Figure()
        temp_plot = np.copy(temp)
        temp_plot[np.isnan(mmp)] = np.nan
        fig.add_trace(
            go.Scatter(
                x=temp_plot,
                y=depths,
                name=self.tr("Temperature"),
                mode="lines",
                line=dict(color="crimson", dash="dash"),
                xaxis="x2",
                connectgaps=False,
            )
        )

        hovertemplate = (
            f"<b>{self.tr('Depth')}: %{{y:.1f}} ft</b><br>MMP: %{{x:.2f}} psia<br>"
            + f"{self.tr('Temperature')}: %{{customdata[0]:.2f}} °F<br>{self.tr('API Gravity')}: %{{customdata[1]:.2f}}°<extra></extra>"
        )
        fig.add_trace(
            go.Scatter(
                x=mmp,
                y=depths,
                name="MMP",
                mode="lines",
                line=dict(color="royalblue", width=3, shape="hv"),
                customdata=np.stack((temp, api), axis=-1),
                hovertemplate=hovertemplate,
                connectgaps=False,
            )
        )

        fig.update_layout(
            title=self.tr("MMP & Temperature Profile for Well: <b>{well_name}</b>").format(
                well_name=well_name
            ),
            xaxis_title=self.tr("MMP (psia)"),
            yaxis_title=self.tr("Depth (ft)"),
            yaxis=dict(autorange="reversed"),
            xaxis2=dict(
                title=self.tr("Temperature (°F)"), overlaying="x", side="top", showgrid=False
            ),
            legend=dict(
                yanchor="top", y=0.98, xanchor="left", x=0.02, bgcolor="rgba(255,255,255,0.7)"
            ),
            margin=dict(t=80),
        )
        return fig

    def update_graphs(self):
        """Public method to be called after loading a project to refresh all graphs."""
        logger.info("OptimizationWidget explicitly requested to update all graphs.")
        if self.current_results:
            self._setup_analysis_options()
            self._generate_selected_plot()
            self._display_detailed_summary(self.current_results)
            self._display_summary(self.current_results)
            self._display_dynamic_table(self.current_results)
        if self.mmp_worker and self.mmp_worker.result:
            self._on_mmp_result(self.mmp_worker.result)
