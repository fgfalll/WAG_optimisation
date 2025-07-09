import logging
from typing import Optional, Any, Dict, List
from dataclasses import fields, is_dataclass

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QComboBox, QPushButton, QTableWidget, QTableWidgetItem, QHeaderView,
    QSplitter, QAbstractItemView, QMessageBox, QSpinBox, QDoubleSpinBox, QFormLayout,
    QSizePolicy, QTabWidget, QGridLayout, QPlainTextEdit, QDialog, QCheckBox, QDialogButtonBox
)
from PyQt6.QtGui import QIcon, QTextOption
from PyQt6.QtCore import Qt, pyqtSignal, QObject, pyqtSlot
import pandas as pd
from PyQt6.QtWebEngineWidgets import QWebEngineView
import plotly.graph_objects as go

from .workers.optimization_worker import OptimizationWorker
from co2eor_optimizer.core.optimisation_engine import OptimizationEngine
from co2eor_optimizer.core.data_models import GeneticAlgorithmParams, EconomicParameters, BayesianOptimizationParams

logger = logging.getLogger(__name__)

# UI Definitions
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

# Parameter Metadata
PARAMETER_METADATA = {
    "pressure": {"display": "Injection Pressure", "unit": "(psi)"}, "rate": {"display": "Injection Rate", "unit": "(bpd)"},
    "v_dp_coefficient": {"display": "V_DP Coefficient", "unit": "(dim.)"}, "mobility_ratio": {"display": "Mobility Ratio", "unit": "(dim.)"},
    "cycle_length_days": {"display": "WAG Cycle Length", "unit": "(days)"}, "water_fraction": {"display": "WAG Water Fraction", "unit": "(dim.)"},
    "oil_price_usd_per_bbl": {"display": "Oil Price", "unit": "($/bbl)"}, "co2_purchase_cost_usd_per_tonne": {"display": "CO2 Purchase Cost", "unit": "($/tonne)"},
    "co2_recycle_cost_usd_per_tonne": {"display": "CO2 Recycle Cost", "unit": "($/tonne)"}, "co2_injection_cost_usd_per_mscf": {"display": "CO2 Injection Cost", "unit": "($/MSCF)"},
    "water_injection_cost_usd_per_bbl": {"display": "Water Injection Cost", "unit": "($/bbl)"}, "water_disposal_cost_usd_per_bbl": {"display": "Water Disposal Cost", "unit": "($/bbl)"},
    "discount_rate_fraction": {"display": "Discount Rate", "unit": "(fraction)"}, "operational_cost_usd_per_bbl_oil": {"display": "OPEX per Barrel", "unit": "($/bbl oil)"},
    "population_size": {"display": "Population Size", "unit": ""}, "generations": {"display": "Generations", "unit": ""},
    "crossover_rate": {"display": "Crossover Rate", "unit": ""}, "mutation_rate": {"display": "Mutation Rate", "unit": ""},
    "elite_count": {"display": "Elite Count", "unit": ""}, "tournament_size": {"display": "Tournament Size", "unit": ""},
    "blend_alpha_crossover": {"display": "Blend Alpha", "unit": ""}, "mutation_strength_factor": {"display": "Mutation Strength", "unit": ""},
    "n_iterations": {"display": "BO Iterations", "unit": ""}, "n_initial_points": {"display": "BO Random Initial Points", "unit": ""},
    "porosity": {"display": "Reservoir Porosity", "unit": "(v/v)"}, "permeability": {"display": "Reservoir Permeability", "unit": "(mD)"},
    "ooip_stb": {"display": "Original Oil In Place", "unit": "(STB)"}
}

class UnlockParametersDialog(QDialog):
    def __init__(self, available_params: List[str], parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("Unlock Parameters for Re-Optimization")
        self.setMinimumWidth(400)
        
        layout = QVBoxLayout(self)
        info_label = QLabel("The optimizer could not reach the target with the current constraints.\n"
                            "Select which underlying reservoir or economic parameters to 'unlock', allowing the optimizer to vary them to meet the target.")
        info_label.setWordWrap(True)
        layout.addWidget(info_label)
        
        self.checkboxes: Dict[str, QCheckBox] = {}
        form_layout = QFormLayout()
        
        for param_key in available_params:
            meta = PARAMETER_METADATA.get(param_key, {})
            display_name = meta.get("display", param_key.replace("_", " ").title())
            checkbox = QCheckBox(display_name)
            self.checkboxes[param_key] = checkbox
            form_layout.addRow(checkbox)
            
        layout.addLayout(form_layout)
        
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        button_box.button(QDialogButtonBox.StandardButton.Ok).setText("Re-run Optimization")
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def get_selected_parameters(self) -> List[str]:
        return [key for key, checkbox in self.checkboxes.items() if checkbox.isChecked()]

class QLogSignalHandler(logging.Handler):
    class _Emitter(QObject):
        log_signal = pyqtSignal(str)
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.emitter = self._Emitter()
    def emit(self, record: logging.LogRecord):
        msg = self.format(record)
        self.emitter.log_signal.emit(msg)


class OptimizationWidget(QWidget):
    optimization_completed = pyqtSignal(dict)
    # NEW SIGNAL: To request the main window to open the main configuration panel.
    open_configuration_requested = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.engine: Optional[OptimizationEngine] = None
        self.worker: Optional[OptimizationWorker] = None
        self.current_results: Optional[Dict[str, Any]] = None
        self.ga_param_inputs: Dict[str, QWidget] = {}
        self.bo_param_inputs: Dict[str, QWidget] = {}
        # Economic param inputs are no longer needed here.
        self.ga_live_data: List[Dict[str, float]] = []

        self._initial_setup_complete = False
        self._setup_logging()
        self._setup_ui()
        self._connect_signals()

        if not all([OptimizationWorker, OptimizationEngine, GeneticAlgorithmParams, EconomicParameters]):
            self.setEnabled(False)
            QMessageBox.critical(self, "Component Error", "Core components for optimization are missing. This tab will be disabled.")
        self._initial_setup_complete = True
    
    def _setup_logging(self):
        self.log_handler = QLogSignalHandler()
        formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s', datefmt='%H:%M:%S')
        self.log_handler.setFormatter(formatter)
        logging.getLogger('co2eor_optimizer').addHandler(self.log_handler)
        logging.getLogger('co2eor_optimizer').setLevel(logging.INFO)

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        config_panel = self._create_config_panel()
        results_panel = self._create_results_panel()
        splitter.addWidget(config_panel)
        splitter.addWidget(results_panel)
        splitter.setSizes([450, 850])
        main_layout.addWidget(splitter, 1)
        self.status_label = QLabel("<i>Ready to start optimization.</i>")
        self.status_label.setWordWrap(True)
        main_layout.addWidget(self.status_label)
        log_group = QGroupBox("Optimization Engine Log")
        log_layout = QVBoxLayout(log_group)
        self.log_display = QPlainTextEdit()
        self.log_display.setReadOnly(True)
        self.log_display.setWordWrapMode(QTextOption.WrapMode.NoWrap)
        font = self.log_display.font()
        font.setFamily("Courier New"); font.setPointSize(10)
        self.log_display.setFont(font); self.log_display.setMaximumHeight(200)
        log_layout.addWidget(self.log_display)
        main_layout.addWidget(log_group)

    def _create_config_panel(self) -> QWidget:
        container = QWidget(); layout = QVBoxLayout(container)
        setup_group = QGroupBox("Optimization Setup")
        setup_grid = QGridLayout(setup_group)
        self.method_combo = QComboBox(); self.method_combo.addItems(OPTIMIZATION_METHODS.keys())
        self.objective_combo = QComboBox(); self.objective_combo.addItems(OPTIMIZATION_OBJECTIVES.keys())
        setup_grid.addWidget(QLabel("Method:"), 0, 0); setup_grid.addWidget(self.method_combo, 0, 1)
        setup_grid.addWidget(QLabel("Objective:"), 1, 0); setup_grid.addWidget(self.objective_combo, 1, 1)
        layout.addWidget(setup_group)

        # Economic parameters group is removed from this widget.
        
        self.ga_params_group = QGroupBox("Genetic Algorithm Parameters")
        self.ga_params_form = QFormLayout(self.ga_params_group); self.ga_params_group.setVisible(False)
        layout.addWidget(self.ga_params_group)
        
        self.bo_params_group = QGroupBox("Bayesian Optimization Parameters")
        self.bo_params_form = QFormLayout(self.bo_params_group); self.bo_params_group.setVisible(False)
        layout.addWidget(self.bo_params_group)
        
        layout.addStretch()

        # Action buttons layout
        action_button_layout = QHBoxLayout()
        self.configure_button = QPushButton(QIcon.fromTheme("document-properties"), "Configure Parameters...")
        self.configure_button.setToolTip("Open the main configuration panel for economic, operational, and other settings.")
        self.run_button = QPushButton(QIcon.fromTheme("system-run"), "Run Optimization")
        self.run_button.setEnabled(False); self.run_button.setToolTip("Load a project to enable the optimization engine.")
        
        action_button_layout.addWidget(self.configure_button)
        action_button_layout.addStretch()
        action_button_layout.addWidget(self.run_button)
        layout.addLayout(action_button_layout)
        
        return container

    def _create_results_panel(self) -> QWidget:
        container = QWidget(); layout = QVBoxLayout(container)
        self.results_tabs = QTabWidget()
        input_summary_tab = QWidget(); input_summary_layout = QVBoxLayout(input_summary_tab)
        self.input_params_table = QTableWidget(); self.input_params_table.setColumnCount(2)
        self.input_params_table.setHorizontalHeaderLabels(["Parameter", "Value"]); self.input_params_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.input_params_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.input_params_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        input_summary_layout.addWidget(self.input_params_table)
        summary_tab = QWidget(); summary_layout = QVBoxLayout(summary_tab)
        self.summary_label = QLabel("<i>Run optimization to view results.</i>"); self.summary_label.setWordWrap(True)
        summary_layout.addWidget(self.summary_label)
        self.results_table = QTableWidget(); self.results_table.setColumnCount(3)
        self.results_table.setHorizontalHeaderLabels(["Parameter", "Optimized Value", "Description"])
        self.results_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.results_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.results_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.results_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        summary_layout.addWidget(self.results_table)
        ga_plot_tab = QWidget(); ga_plot_layout = QVBoxLayout(ga_plot_tab)
        self.ga_plot_view = QWebEngineView(); ga_plot_layout.addWidget(self.ga_plot_view)
        analysis_tab = QWidget(); analysis_layout = QVBoxLayout(analysis_tab)
        analysis_group = QGroupBox("Analysis Plot Generator"); analysis_controls_layout = QHBoxLayout(analysis_group)
        self.plot_type_combo = QComboBox(); self.sensitivity_param_combo = QComboBox()
        self.show_analysis_plot_button = QPushButton("Generate Plot")
        analysis_controls_layout.addWidget(QLabel("Plot Type:")); analysis_controls_layout.addWidget(self.plot_type_combo, 1)
        analysis_controls_layout.addWidget(self.sensitivity_param_combo, 2); analysis_controls_layout.addWidget(self.show_analysis_plot_button, 0)
        self.analysis_plot_view = QWebEngineView(); self.analysis_plot_view.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        analysis_layout.addWidget(analysis_group); analysis_layout.addWidget(self.analysis_plot_view, 1)
        self.results_tabs.addTab(input_summary_tab, "Input Parameters"); self.results_tabs.addTab(summary_tab, "Results Summary")
        self.results_tabs.addTab(ga_plot_tab, "GA Progress"); self.results_tabs.addTab(analysis_tab, "Analysis Plots")
        layout.addWidget(self.results_tabs)
        return container

    def _connect_signals(self):
        self.method_combo.currentTextChanged.connect(self._on_method_changed)
        self.run_button.clicked.connect(self._run_optimization)
        self.configure_button.clicked.connect(self.open_configuration_requested.emit)
        self.show_analysis_plot_button.clicked.connect(self._run_analysis_plot)
        self.plot_type_combo.currentTextChanged.connect(self._on_analysis_plot_type_changed)
        self.log_handler.emitter.log_signal.connect(self._append_log_message)

    @pyqtSlot(str)
    def _append_log_message(self, message: str):
        self.log_display.appendPlainText(message)

    def update_engine(self, engine: Optional[OptimizationEngine]):
        self.engine = engine; is_engine_ready = self.engine is not None
        self.run_button.setEnabled(is_engine_ready); self.run_button.setToolTip("Run the configured optimization." if is_engine_ready else "Load a project to enable.")
        if self.engine:
            # We no longer populate the economic form here.
            self._populate_dataclass_form(self.engine.ga_params_default_config, self.ga_params_form, self.ga_param_inputs)
            self._populate_dataclass_form(self.engine.bo_params_default_config, self.bo_params_form, self.bo_param_inputs)
            self._on_method_changed(self.method_combo.currentText()); logger.info("OptimizationWidget engine instance updated and inputs displayed.")
        else:
            self.ga_params_group.setVisible(False); self.bo_params_group.setVisible(False)
            self._clear_form(self.ga_params_form, self.ga_param_inputs)
            self._clear_form(self.bo_params_form, self.bo_param_inputs); self._update_input_summary_tab(); logger.warning("OptimizationWidget engine instance removed.")

    @pyqtSlot(dict)
    def on_configurations_updated(self, full_config_data: Dict[str, Any]):
        """Receives the full configuration dictionary and updates the relevant parameter forms."""
        logger.info("OptimizationWidget received configuration update. Refreshing forms.")
        if not self._initial_setup_complete:
            return

        for dc_name, dc_instance in full_config_data.items():
            if dc_name == GeneticAlgorithmParams.__name__:
                self._populate_dataclass_form(dc_instance, self.ga_params_form, self.ga_param_inputs)
            elif dc_name == BayesianOptimizationParams.__name__:
                self._populate_dataclass_form(dc_instance, self.bo_params_form, self.bo_param_inputs)
        # Refresh the summary tab to reflect any changes (e.g., in economic parameters).
        self._update_input_summary_tab()

    def _on_method_changed(self, method_name: str):
        method_name_lower = method_name.lower()
        show_ga = "genetic" in method_name_lower or "hybrid" in method_name_lower
        show_bo = "bayesian" in method_name_lower or "hybrid" in method_name_lower
        self.ga_params_group.setVisible(show_ga); self.bo_params_group.setVisible(show_bo)
        self.results_tabs.setTabEnabled(2, show_ga)
        if self._initial_setup_complete: self._update_input_summary_tab()

    def _clear_form(self, form_layout: QFormLayout, input_dict: Dict):
        while form_layout.rowCount() > 0: form_layout.removeRow(0)
        input_dict.clear()

    def _populate_dataclass_form(self, dc_instance: Any, form_layout: QFormLayout, input_dict: Dict):
        self._clear_form(form_layout, input_dict)
        if not (dc_instance and is_dataclass(dc_instance)): return
        for field in fields(dc_instance):
            meta = PARAMETER_METADATA.get(field.name, {}); display_name = meta.get("display", field.name.replace("_", " ").title()); unit = meta.get("unit", ""); description = meta.get("description", "No description available.")
            label_text = f"{display_name} {unit}".strip(); label_widget = QLabel(label_text); label_widget.setToolTip(description)
            value = getattr(dc_instance, field.name); widget = None
            if field.type is int: widget = QSpinBox(); widget.setRange(-1_000_000, 1_000_000); widget.setValue(value)
            elif field.type is float: widget = QDoubleSpinBox(); widget.setRange(-1_000_000.0, 1_000_000.0); widget.setDecimals(4); widget.setValue(value)
            if widget: widget.valueChanged.connect(self._update_input_summary_tab); input_dict[field.name] = widget; form_layout.addRow(label_widget, widget)
        self._update_input_summary_tab()

    def _get_params_from_form(self, dc_class: type, input_dict: Dict) -> Optional[Any]:
        if not dc_class: return None
        try: kwargs = {name: w.value() for name, w in input_dict.items()}; return dc_class(**kwargs)
        except Exception as e: QMessageBox.critical(self, "Settings Error", f"Invalid parameter input for {dc_class.__name__}: {e}"); logger.error(f"Error creating {dc_class.__name__} from form: {e}", exc_info=True); return None

    def _update_input_summary_tab(self):
        self.input_params_table.setRowCount(0)
        if not self.engine: return
        params = {}; params["MMP (Calculated, psi)"] = f"{self.engine.mmp:.0f}" if self.engine.mmp else "N/A"; params["Average Porosity"] = f"{self.engine.avg_porosity:.3f}"; params["OOIP (STB)"] = f"{self.engine.reservoir.ooip_stb:,.0f}"; params["Project Lifetime (years)"] = self.engine.operational_params.project_lifetime_years
        for f in fields(self.engine.eor_params): meta = PARAMETER_METADATA.get(f.name, {}); display_name = meta.get("display", f.name.replace('_', ' ').title()); key = f"EOR: {display_name}"; params[key] = getattr(self.engine.eor_params, f.name)
        # Economic parameters are now read-only from the engine's state
        for f in fields(self.engine.economic_params): meta = PARAMETER_METADATA.get(f.name, {}); display_name = meta.get("display", f.name.replace('_', ' ').title()); key = f"Econ: {display_name}"; params[key] = getattr(self.engine.economic_params, f.name)
        if self.ga_params_group.isVisible():
            for f in fields(self.engine.ga_params_default_config): meta = PARAMETER_METADATA.get(f.name, {}); display_name = meta.get("display", f.name.replace('_', ' ').title()); key = f"GA: {display_name}"; params[key] = getattr(self.engine.ga_params_default_config, f.name)
        if self.bo_params_group.isVisible():
            for f in fields(self.engine.bo_params_default_config): meta = PARAMETER_METADATA.get(f.name, {}); display_name = meta.get("display", f.name.replace('_', ' ').title()); key = f"BO: {display_name}"; params[key] = getattr(self.engine.bo_params_default_config, f.name)
        self.input_params_table.setRowCount(len(params))
        for i, (key, val) in enumerate(sorted(params.items())): val_str = f"{val:,.4g}" if isinstance(val, (float, int)) else str(val); self.input_params_table.setItem(i, 0, QTableWidgetItem(key)); self.input_params_table.setItem(i, 1, QTableWidgetItem(val_str))
        self.input_params_table.resizeRowsToContents()

    def _run_optimization(self):
        if not self.engine: QMessageBox.critical(self, "Engine Error", "Optimization Engine is not available."); return
        if self.worker and self.worker.isRunning(): QMessageBox.warning(self, "Busy", "An optimization run is already in progress."); return
        self._clear_results(clear_inputs=False)
        method_key = self.method_combo.currentText(); method_name = OPTIMIZATION_METHODS[method_key]
        self.engine.chosen_objective = OPTIMIZATION_OBJECTIVES[self.objective_combo.currentText()]
        
        # Economic params are no longer read from a local form. The engine's state is used directly.
        
        kwargs = {}
        if self.ga_params_group.isVisible():
            ga_params = self._get_params_from_form(GeneticAlgorithmParams, self.ga_param_inputs)
            if not ga_params: return
            kwargs["ga_params_override"] = ga_params
        if self.bo_params_group.isVisible():
            bo_params = self._get_params_from_form(BayesianOptimizationParams, self.bo_param_inputs)
            if not bo_params: return
            kwargs["bo_params_override"] = bo_params
            
        self.results_tabs.setCurrentIndex(0)
        self.run_button.setEnabled(False); self.status_label.setText("<i>Starting optimization...</i>")
        self.worker = OptimizationWorker(self.engine, method_name, kwargs)
        self.worker.result_ready.connect(self._on_result)
        self.worker.error_occurred.connect(self._on_error)
        self.worker.finished.connect(self._on_worker_finished)
        self.worker.progress_updated.connect(self._on_progress_update)
        self.worker.ga_progress_updated.connect(self._update_ga_live_plot)
        self.worker.target_unreachable.connect(self._handle_target_unreachable)
        self.worker.start()

    def _clear_results(self, clear_inputs=True):
        self.summary_label.setText("<i>Run optimization to view results.</i>")
        self.results_table.setRowCount(0)
        if clear_inputs: self.input_params_table.setRowCount(0)
        self.analysis_plot_view.setHtml(""), self.ga_plot_view.setHtml(""), self.log_display.clear()
        self.results_tabs.setTabEnabled(3, False)
        self.current_results = None; self.ga_live_data.clear()

    def _on_progress_update(self, message: str):
        self.status_label.setText(f"<i>{message}</i>")

    def _on_result(self, results: Dict[str, Any]):
        self.current_results = results; self._display_summary(results); self._display_dynamic_table(results)
        self.optimization_completed.emit(results); self._setup_analysis_options(); self.results_tabs.setCurrentIndex(1)

    def _display_dynamic_table(self, results: Dict[str, Any]):
        params_to_show = results.get('optimized_params_final_clipped', {}).copy()
        self.results_table.setRowCount(0); self.results_table.setRowCount(len(params_to_show))
        sorted_params = sorted(params_to_show.items())
        for i, (key, val) in enumerate(sorted_params):
            meta = PARAMETER_METADATA.get(key, {})
            display_name = meta.get("display", key.replace("_", " ").title()); unit = meta.get("unit", "")
            description = "Operational parameter optimized within standard bounds."
            if self.engine and key in self.engine.RELAXABLE_CONSTRAINTS:
                 description = self.engine.RELAXABLE_CONSTRAINTS[key].get('description', 'Underlying parameter.')
                 if key in results.get('unlocked_params_in_run', []):
                     description += " (This parameter was unlocked and optimized to meet the target.)"
            val_str = f"{val:,.4g}" if isinstance(val, (float, int)) else str(val)
            self.results_table.setItem(i, 0, QTableWidgetItem(f"{display_name} {unit}".strip()))
            self.results_table.setItem(i, 1, QTableWidgetItem(val_str))
            self.results_table.setItem(i, 2, QTableWidgetItem(description))
        self.results_table.resizeRowsToContents(); self.results_table.resizeColumnToContents(0); self.results_table.resizeColumnToContents(1)

    def _display_summary(self, results: Dict[str, Any]):
        obj_val = results.get('objective_function_value', 'N/A'); obj_name = results.get('chosen_objective', 'N/A').replace("_", " ").title()
        rf = results.get('final_recovery_factor_reported', 'N/A'); method = results.get('method', 'N/A').replace("_", " ").title()
        obj_val_str = f"{obj_val:,.4g}" if isinstance(obj_val, (float, int)) else str(obj_val)
        rf_str = f"{rf:.4f}" if isinstance(rf, (float, int)) else str(rf)
        summary = (f"<b>Optimization Complete: {method}</b><br>"f"Final Objective ({obj_name}): <b>{obj_val_str}</b><br>"f"Final Recovery Factor: <b>{rf_str}</b>")
        self.summary_label.setText(summary)

    @pyqtSlot(dict)
    def _update_ga_live_plot(self, progress_data: Dict[str, Any]):
        self.ga_live_data.append(progress_data)
        generations = [d['generation'] for d in self.ga_live_data]; best_fitness = [d['best_fitness'] for d in self.ga_live_data]
        avg_fitness = [d['avg_fitness'] for d in self.ga_live_data]; worst_fitness = [d['worst_fitness'] for d in self.ga_live_data]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=generations, y=best_fitness, mode='lines', name='Best Fitness'))
        fig.add_trace(go.Scatter(x=generations, y=avg_fitness, mode='lines', name='Average Fitness', line=dict(dash='dot')))
        fig.add_trace(go.Scatter(x=generations, y=worst_fitness, mode='lines', name='Worst Fitness', line=dict(dash='dash'), opacity=0.5))
        fig.update_layout(title_text='Live Genetic Algorithm Progress', xaxis_title_text='Generation', yaxis_title_text='Fitness (Objective Value)', legend_title_text='Metric')
        self.ga_plot_view.setHtml(fig.to_html(include_plotlyjs='cdn'))

    def _setup_analysis_options(self):
        self.results_tabs.setTabEnabled(3, True); self.plot_type_combo.clear()
        plot_options = ["Convergence"]
        if self.current_results: plot_options.append("Parameter Sensitivity")
        if self.engine and self.engine.well_analysis: plot_options.append("MMP Profile")
        self.plot_type_combo.addItems(plot_options); self._on_analysis_plot_type_changed(self.plot_type_combo.currentText())
        
    def _on_analysis_plot_type_changed(self, plot_type: str):
        show_sensitivity_combo = (plot_type == "Parameter Sensitivity")
        self.sensitivity_param_combo.setVisible(show_sensitivity_combo)
        if show_sensitivity_combo: self._populate_sensitivity_combo()

    def _populate_sensitivity_combo(self):
        self.sensitivity_param_combo.clear()
        if not self.current_results: return
        params = list(self.current_results.get('optimized_params_final_clipped', {}).keys())
        econ_params = [f.name for f in EconomicParameters.__dataclass_fields__]
        all_param_keys = sorted(list(set(params + econ_params)))
        for key in all_param_keys:
            meta = PARAMETER_METADATA.get(key, {}); display_name = meta.get("display", key.replace("_", " ").title())
            self.sensitivity_param_combo.addItem(display_name, userData=key)

    def _run_analysis_plot(self):
        if not self.engine: return
        plot_type = self.plot_type_combo.currentText(); fig = None
        try:
            if plot_type == "Convergence" and self.current_results: fig = self.engine.plot_optimization_convergence(self.current_results)
            elif plot_type == "MMP Profile": fig = self.engine.plot_mmp_profile()
            elif plot_type == "Parameter Sensitivity" and self.current_results:
                param_key = self.sensitivity_param_combo.currentData()
                if param_key: fig = self.engine.plot_parameter_sensitivity(param_key, self.current_results)
            if fig: self.analysis_plot_view.setHtml(fig.to_html(include_plotlyjs='cdn'))
            else: QMessageBox.information(self, "Plot Error", f"Could not generate plot for '{plot_type}'.")
        except Exception as e:
            logger.error(f"Error generating analysis plot '{plot_type}': {e}", exc_info=True)
            QMessageBox.critical(self, "Plotting Error", f"An error occurred while generating the plot: {e}")

    @pyqtSlot(dict)
    def _handle_target_unreachable(self, failed_results: dict):
        if not self.engine: return
        self.status_label.setText("<b style='color:orange;'>Target was not reached with current constraints.</b>")
        self._on_result(failed_results)
        target_name = failed_results.get('target_objective_name_in_run', 'target').replace('_', ' ')
        achieved_val = failed_results.get('final_target_value_achieved', 0.0)
        target_val = failed_results.get('target_objective_value_in_run', 0.0)
        reply = QMessageBox.information(self, "Target Unreachable",
            f"The optimizer could not meet the specified {target_name} target of {target_val:.4f}.\n\n"
            f"The closest value found was {achieved_val:.4f}.\n\n"
            "Would you like to unlock additional reservoir or economic parameters to allow the optimizer more freedom to meet the target?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.Yes)
        if reply == QMessageBox.StandardButton.Yes:
            available_params = list(self.engine.RELAXABLE_CONSTRAINTS.keys())
            dialog = UnlockParametersDialog(available_params, self)
            if dialog.exec():
                unlocked_params = dialog.get_selected_parameters()
                if unlocked_params:
                    logger.info(f"User chose to unlock parameters for re-run: {unlocked_params}")
                    self._rerun_optimization_with_unlocked_params(unlocked_params)
                else:
                    self.status_label.setText("<i>Re-run cancelled. No parameters were unlocked.</i>")
    
    def _rerun_optimization_with_unlocked_params(self, unlocked_params: List[str]):
        if not self.engine: return
        logger.info(f"Preparing for re-optimization with unlocked parameters: {unlocked_params}")
        self.engine.prepare_for_rerun_with_unlocked_params(unlocked_params)
        self._clear_results(clear_inputs=False)
        method_key = self.method_combo.currentText(); method_name = OPTIMIZATION_METHODS[method_key]
        kwargs = {"ga_params_override": self._get_params_from_form(GeneticAlgorithmParams, self.ga_param_inputs),
                  "bo_params_override": self._get_params_from_form(BayesianOptimizationParams, self.bo_param_inputs)}
        self.run_button.setEnabled(False)
        self.status_label.setText(f"<i>Re-running with unlocked parameters: {', '.join(unlocked_params)}...</i>")
        self.worker = OptimizationWorker(self.engine, method_name, kwargs)
        self.worker.result_ready.connect(self._on_result)
        self.worker.target_unreachable.connect(self._on_result)
        self.worker.error_occurred.connect(self._on_error)
        self.worker.finished.connect(self._on_worker_finished)
        self.worker.progress_updated.connect(self._on_progress_update)
        self.worker.ga_progress_updated.connect(self._update_ga_live_plot)
        self.worker.start()

    def _on_error(self, error_msg: str):
        self._clear_results()
        self.status_label.setText(f"<p style='color:red;'><b>Error:</b> {error_msg}</p>")
        self.run_button.setEnabled(self.engine is not None)

    def _on_worker_finished(self):
        self.run_button.setEnabled(self.engine is not None)
        if self.engine:
            self.engine.reset_to_base_state()
            logger.info("Engine state reset to base configuration after worker finished.")

        if self.worker and not self.worker.was_successful():
             if "Error" not in self.status_label.text():
                 self.status_label.setText("<i>Optimization worker finished with issues. Check log.</i>")
        elif self.current_results:
             if self.current_results.get('target_was_unreachable'):
                 self.status_label.setText("<i><b style='color:orange;'>Re-optimization finished. Target may still be unreachable.</b></i>")
             else:
                 self.status_label.setText("<i>Optimization finished successfully.</i>")
        else:
            if "Error" not in self.status_label.text():
                 self.status_label.setText("<i>Optimization worker finished. Ready for next run.</i>")

        if self.worker:
            self.worker.deleteLater()
            self.worker = None