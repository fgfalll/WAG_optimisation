import logging
import sys
import os
import numpy as np
from pathlib import Path
from typing import Optional, Any, Dict, List, Union
from copy import deepcopy

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QStatusBar, QMessageBox,
    QFileDialog, QTabWidget, QApplication, QSplitter, QLabel, QProgressBar
)
from PyQt6.QtGui import QAction, QIcon, QKeySequence, QCloseEvent
from PyQt6.QtCore import Qt, pyqtSignal, QSettings, QSize, QTimer
import traceback

# --- Resource Monitoring Imports ---
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logging.warning("psutil library not found. CPU/RAM monitoring will be disabled.")

try:
    import GPUtil
    GPUTIL_AVAILABLE = True
except ImportError:
    GPUTIL_AVAILABLE = False
    logging.warning("GPUtil library not found. GPU monitoring will be disabled.")


# --- Setup Logger ---
logger = logging.getLogger(__name__)

# --- Project Imports ---
try:
    from co2eor_optimizer.config_manager import config_manager, ConfigManager
    from co2eor_optimizer.utils.project_file_handler import save_project_to_tphd, load_project_from_tphd
    from co2eor_optimizer.utils.units_manager import units_manager, UnitsManager
    from co2eor_optimizer.utils.report_generator import ReportGenerator

    from co2eor_optimizer.ui.overview_page import OverviewPageWidget
    from co2eor_optimizer.ui.data_management_widget import DataManagementWidget
    from co2eor_optimizer.ui.config_widget import ConfigWidget
    from co2eor_optimizer.ui.mmp_well_analysis_widget import MMPWellAnalysisWidget
    from co2eor_optimizer.ui.optimization_widget import OptimizationWidget
    from co2eor_optimizer.ui.sensitivity_widget import SensitivityWidget
    from co2eor_optimizer.ui.uq_widget import UQWidget
    from co2eor_optimizer.ui.ai_assistant_widget import AIAssistantWidget
    from co2eor_optimizer.ui.logging_widget import LoggingWidget
    from co2eor_optimizer.ui.dialogs.report_config_dialog import ReportConfigDialog

    from co2eor_optimizer.core.data_models import (
        WellData, ReservoirData, PVTProperties,
        EconomicParameters, EORParameters, OperationalParameters,
        ProfileParameters, GeneticAlgorithmParams
    )
    from co2eor_optimizer.core.optimisation_engine import OptimizationEngine
    from co2eor_optimizer.analysis.sensitivity_analyzer import SensitivityAnalyzer
    from co2eor_optimizer.analysis.uq_engine import UncertaintyQuantificationEngine
    from co2eor_optimizer.analysis.well_analysis import WellAnalysis
    
    logger.info("All MainWindow project modules imported successfully.")

except ImportError as e_mw_imp:
    log_msg = f"FATAL ERROR in MainWindow: Could not import one or more required modules: {e_mw_imp}. Application cannot continue."
    logger.critical(log_msg, exc_info=True)
    
    app = QApplication.instance()
    if app:
        error_msg = QMessageBox()
        error_msg.setIcon(QMessageBox.Icon.Critical)
        error_msg.setWindowTitle("Critical Import Error")
        error_msg.setText(f"FATAL: Could not import required modules")
        error_msg.setDetailedText(f"{e_mw_imp}\n\nTraceback:\n{traceback.format_exc()}")
        error_msg.exec()
    else:
        print(log_msg, file=sys.stderr)
        print(f"Traceback:\n{traceback.format_exc()}", file=sys.stderr)
    sys.exit(1)


class MainWindow(QMainWindow):
    """
    The main application window for the CO2 EOR Suite.
    Manages the overall UI structure, project data flow, and interactions
    between different modules.
    """
    project_file_path: Optional[Path] = None
    is_project_modified: bool = False
    active_unit_system: str = "Field"

    # --- FIX: Renamed for clarity and consistency ---
    current_well_data: List[WellData]
    current_reservoir_data: Optional[ReservoirData] # Typically one merged reservoir model
    current_pvt_properties: Optional[PVTProperties]
    
    current_economic_params: EconomicParameters
    current_eor_params: EORParameters
    current_operational_params: OperationalParameters
    current_profile_params: ProfileParameters
    current_ga_params: GeneticAlgorithmParams

    optimisation_engine_instance: Optional[OptimizationEngine] = None
    sensitivity_analyzer_instance: Optional[SensitivityAnalyzer] = None
    # uq_engine_instance: Optional[UncertaintyQuantificationEngine] = None

    def __init__(self, app_settings: QSettings, config_manager_instance: ConfigManager):
        super().__init__(None)
        self.app_settings = app_settings
        self.config_manager = config_manager_instance
        
        self._initialize_project_data_and_configs()
        self.report_generator = ReportGenerator(units_manager)
        
        self._setup_ui_structure()
        self._create_actions()
        self._create_menu_bar()
        self._create_status_bar()
        self._create_tool_bar()
        logger.info("MainWindow UI structure, actions, and menus initialized.")

        self._setup_initial_view()
        self.load_window_settings()
        self._update_window_title()

        # --- ENHANCEMENT: Setup resource monitor timer ---
        self.resource_monitor_timer = QTimer(self)
        self.resource_monitor_timer.setInterval(2000) # Update every 2 seconds
        self.resource_monitor_timer.timeout.connect(self._update_resource_monitors)
        self.resource_monitor_timer.start()

    def _initialize_project_data_and_configs(self, from_project_load: bool = False):
        logger.info(f"Initializing project data and configurations (from_project_load={from_project_load}).")
        self.current_well_data = []
        self.current_reservoir_data = None
        self.current_pvt_properties = None

        if not from_project_load:
            try:
                self.current_economic_params = EconomicParameters.from_config_dict(self.config_manager.get_section("EconomicParametersDefaults") or {})
                self.current_eor_params = EORParameters.from_config_dict(self.config_manager.get_section("EORParametersDefaults") or {})
                self.current_operational_params = OperationalParameters.from_config_dict(self.config_manager.get_section("OperationalParametersDefaults") or {})
                self.current_profile_params = ProfileParameters.from_config_dict(self.config_manager.get_section("ProfileParametersDefaults") or {})
                self.current_ga_params = GeneticAlgorithmParams.from_config_dict(self.config_manager.get_section("GeneticAlgorithmParamsDefaults") or {})
            except Exception as e:
                logger.critical(f"Failed to initialize one or more configuration dataclasses from defaults. Error: {e}", exc_info=True)
                raise
        
        self.optimisation_engine_instance = None
        self.sensitivity_analyzer_instance = None
        self.is_project_modified = False if not from_project_load else self.is_project_modified

    def _setup_ui_structure(self):
        self.setWindowTitle(QApplication.applicationName())
        self.setWindowIcon(QIcon.fromTheme("application-x-executable"))
        self.main_central_widget = QWidget(self)
        self.main_central_layout = QVBoxLayout(self.main_central_widget)
        self.main_central_layout.setContentsMargins(0,0,0,0)
        self.setCentralWidget(self.main_central_widget)
        self._setup_main_app_tabs_container()

    def _setup_initial_view(self):
        self.overview_page = OverviewPageWidget(self)
        self.overview_page.start_new_project_requested.connect(self._handle_start_new_project_action)
        self.overview_page.load_existing_project_requested.connect(self._handle_load_project_action)
        self.overview_page.quick_start_requested.connect(self._handle_quick_start_action)
        self.main_central_layout.addWidget(self.overview_page)
        self.main_tab_widget.setVisible(False)
        self.overview_page.setVisible(True)
        self.save_project_action.setEnabled(False); self.save_project_as_action.setEnabled(False); self.close_project_action.setEnabled(False)

    def _setup_main_app_tabs_container(self):
        self.main_tab_widget = QTabWidget(self); self.main_tab_widget.setTabPosition(QTabWidget.TabPosition.North); self.main_tab_widget.setMovable(True); self.main_tab_widget.setVisible(False)
        self.data_management_tab = DataManagementWidget(self); self.data_management_tab.project_data_updated.connect(self._on_project_data_model_updated); self.data_management_tab.status_message_updated.connect(self.show_status_message); self.main_tab_widget.addTab(self.data_management_tab, QIcon.fromTheme("document-properties"), "1. Data Management")
        self.config_tab = ConfigWidget(self); self.config_tab.configuration_changed.connect(self._on_app_configuration_changed); self.config_tab.save_configuration_to_file_requested.connect(self._on_save_standalone_config_requested); self.main_tab_widget.addTab(self.config_tab, QIcon.fromTheme("preferences-system"), "2. Configuration")
        self.mmp_well_analysis_tab = MMPWellAnalysisWidget(self); self.main_tab_widget.addTab(self.mmp_well_analysis_tab, QIcon.fromTheme("view-plot"), "3. MMP & Well Analysis")
        self.optimization_tab = OptimizationWidget(self); self.optimization_tab.optimization_completed.connect(self._on_optimization_run_completed); self.main_tab_widget.addTab(self.optimization_tab, QIcon.fromTheme("system-run"), "4. Optimization")
        self.sensitivity_tab = SensitivityWidget(self); self.main_tab_widget.addTab(self.sensitivity_tab, QIcon.fromTheme("view-statistics"), "5. Sensitivity Analysis")
        self.uq_tab = UQWidget(self); self.main_tab_widget.addTab(self.uq_tab, QIcon.fromTheme("view-process-users"), "6. Uncertainty Quantification")
        self.ai_assistant_tab = AIAssistantWidget(self.app_settings, self); self.ai_assistant_tab.request_context_data.connect(self._provide_context_to_ai_assistant); self.main_tab_widget.addTab(self.ai_assistant_tab, QIcon.fromTheme("preferences-desktop-ai"), "7. AI Assistant")
        self.logging_tab = LoggingWidget(self); self.main_tab_widget.addTab(self.logging_tab, QIcon.fromTheme("text-x-generic"), "8. Application Logs")
        self.main_central_layout.addWidget(self.main_tab_widget)

    def _transition_to_main_app_view(self, focus_tab_index: int = 0):
        if self.overview_page: self.overview_page.setVisible(False)
        self.main_tab_widget.setVisible(True); self.main_tab_widget.setCurrentIndex(focus_tab_index)
        self.save_project_action.setEnabled(True); self.save_project_as_action.setEnabled(True); self.close_project_action.setEnabled(True); self.new_project_action.setEnabled(True)
        logger.info(f"Transitioned to main app view, tab index: {focus_tab_index}")

    def _create_actions(self):
        self.new_project_action = QAction(QIcon.fromTheme("document-new"), "&New Project", self); self.new_project_action.setShortcut(QKeySequence.StandardKey.New); self.new_project_action.triggered.connect(self._handle_start_new_project_action)
        self.open_project_action = QAction(QIcon.fromTheme("document-open"), "&Open Project...", self); self.open_project_action.setShortcut(QKeySequence.StandardKey.Open); self.open_project_action.triggered.connect(self._handle_load_project_action)
        self.save_project_action = QAction(QIcon.fromTheme("document-save"), "&Save Project", self); self.save_project_action.setShortcut(QKeySequence.StandardKey.Save); self.save_project_action.triggered.connect(self._handle_save_project_action)
        self.save_project_as_action = QAction(QIcon.fromTheme("document-save-as"), "Save Project &As...", self); self.save_project_as_action.setShortcut(QKeySequence.StandardKey.SaveAs); self.save_project_as_action.triggered.connect(self._handle_save_project_as_action)
        self.close_project_action = QAction(QIcon.fromTheme("document-close"), "&Close Project", self); self.close_project_action.triggered.connect(self._handle_close_project_action)
        self.exit_action = QAction(QIcon.fromTheme("application-exit"), "E&xit", self); self.exit_action.setShortcut(QKeySequence.StandardKey.Quit); self.exit_action.triggered.connect(self.close)
        self.settings_action = QAction(QIcon.fromTheme("preferences-system"), "&Preferences...", self); self.settings_action.triggered.connect(self._show_preferences_dialog)
        self.generate_report_action = QAction(QIcon.fromTheme("document-print"), "&Generate Report...", self); self.generate_report_action.triggered.connect(self._show_report_config_dialog)
        self.about_action = QAction(QIcon.fromTheme("help-about"), "&About " + QApplication.applicationName(), self); self.about_action.triggered.connect(self._show_about_dialog)
        self.about_qt_action = QAction("About &Qt", self); self.about_qt_action.triggered.connect(QApplication.aboutQt)

    def _create_menu_bar(self):
        menu_bar = self.menuBar(); file_menu = menu_bar.addMenu("&File"); file_menu.addAction(self.new_project_action); file_menu.addAction(self.open_project_action); file_menu.addSeparator(); file_menu.addAction(self.save_project_action); file_menu.addAction(self.save_project_as_action); file_menu.addSeparator(); file_menu.addAction(self.close_project_action); file_menu.addSeparator(); file_menu.addAction(self.exit_action); edit_menu = menu_bar.addMenu("&Edit"); edit_menu.addAction(self.settings_action); tools_menu = menu_bar.addMenu("&Tools"); tools_menu.addAction(self.generate_report_action); help_menu = menu_bar.addMenu("&Help"); help_menu.addAction(self.about_action); help_menu.addAction(self.about_qt_action)

    def _create_status_bar(self):
        self.status_bar = QStatusBar(self); self.setStatusBar(self.status_bar)
        
        # --- ENHANCEMENT: Add resource monitor labels ---
        if PSUTIL_AVAILABLE:
            self.cpu_label = QLabel(" CPU: --% "); self.status_bar.addPermanentWidget(self.cpu_label)
            self.ram_label = QLabel(" RAM: --% "); self.status_bar.addPermanentWidget(self.ram_label)
        if GPUTIL_AVAILABLE and GPUtil.getGPUs():
            self.gpu_label = QLabel(" GPU: --% "); self.status_bar.addPermanentWidget(self.gpu_label)

        self.status_progress_bar = QProgressBar(self); self.status_progress_bar.setRange(0,100); self.status_progress_bar.setVisible(False); self.status_progress_bar.setFixedSize(150, 16); self.status_bar.addPermanentWidget(self.status_progress_bar)
        self.show_status_message("Ready.", 5000)

    def _update_resource_monitors(self):
        """Updates the CPU, RAM, and GPU labels in the status bar."""
        if PSUTIL_AVAILABLE:
            self.cpu_label.setText(f" CPU: {psutil.cpu_percent():.0f}% ")
            self.ram_label.setText(f" RAM: {psutil.virtual_memory().percent:.0f}% ")
        if GPUTIL_AVAILABLE and hasattr(self, 'gpu_label'):
            try:
                gpu = GPUtil.getGPUs()[0]
                self.gpu_label.setText(f" GPU: {gpu.load*100:.0f}% ")
            except Exception:
                self.gpu_label.setText(" GPU: N/A ")

    def _create_tool_bar(self): pass
    def _update_window_title(self): title = QApplication.applicationName(); title += f" - {self.project_file_path.name}" if self.project_file_path else " - New Project"; self.setWindowTitle(title + "*" if self.is_project_modified else title)
    def set_project_modified(self, modified: bool = True):
        if self.is_project_modified != modified: self.is_project_modified = modified; self._update_window_title()
    def show_status_message(self, message: str, timeout: int = 3000): self.status_bar.showMessage(message, timeout); logger.debug(f"Status: {message}")
    def _handle_start_new_project_action(self):
        if self._confirm_unsaved_changes():
            logger.info("Starting new project."); self.project_file_path = None; self._initialize_project_data_and_configs()
            self.data_management_tab.clear_all_project_data(); self.config_tab.update_configs_from_project({EconomicParameters.__name__: self.current_economic_params, EORParameters.__name__: self.current_eor_params, OperationalParameters.__name__: self.current_operational_params, ProfileParameters.__name__: self.current_profile_params, GeneticAlgorithmParams.__name__: self.current_ga_params})
            self.mmp_well_analysis_tab.update_data([], None); self.optimization_tab.update_engine(None); self.sensitivity_tab.update_analyzer(None)
            self._update_window_title(); self.set_project_modified(False); self._transition_to_main_app_view(focus_tab_index=0); self.show_status_message("New project started. Load data to begin.", 5000)
    def _handle_load_project_action(self):
        if not self._confirm_unsaved_changes(): return
        last_dir = self.app_settings.value("Paths/last_project_dir", str(Path.home())); filepath_str, _ = QFileDialog.getOpenFileName(self, "Open Project File", last_dir, "CO2 EOR Project Files (*.tphd);;All Files (*)")
        if not filepath_str: return
        filepath = Path(filepath_str); self.app_settings.setValue("Paths/last_project_dir", str(filepath.parent)); self.show_status_message(f"Loading project: {filepath.name}...", 0); QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor); QTimer.singleShot(50, lambda: self._perform_project_load(filepath))
    def _perform_project_load(self, filepath: Path):
        try:
            project_data_dict = load_project_from_tphd(filepath);
            if project_data_dict is None: raise IOError("Failed to load or parse project file.")
            self._initialize_project_data_and_configs(from_project_load=True)
            self.current_well_data = project_data_dict.get("well_data_list", [])
            self.current_reservoir_data = project_data_dict.get("reservoir_data", None) # Expect single object
            self.current_pvt_properties = project_data_dict.get("pvt_properties", None)
            self.current_economic_params = project_data_dict.get("economic_parameters", EconomicParameters())
            self.current_eor_params = project_data_dict.get("eor_parameters", EORParameters())
            self.current_operational_params = project_data_dict.get("operational_parameters", OperationalParameters())
            self.current_profile_params = project_data_dict.get("profile_parameters", ProfileParameters())
            self.current_ga_params = project_data_dict.get("ga_parameters", GeneticAlgorithmParams())
            self.data_management_tab.load_data_into_ui(self.current_well_data, self.current_reservoir_data, self.current_pvt_properties)
            self.config_tab.update_configs_from_project({EconomicParameters.__name__: self.current_economic_params, EORParameters.__name__: self.current_eor_params, OperationalParameters.__name__: self.current_operational_params, ProfileParameters.__name__: self.current_profile_params, GeneticAlgorithmParams.__name__: self.current_ga_params})
            self._reinitialize_engines_and_analysis_tabs()
            self.project_file_path = filepath; self.set_project_modified(False); self._update_window_title(); self._transition_to_main_app_view(focus_tab_index=0); self.show_status_message(f"Project '{filepath.name}' loaded successfully.", 5000); logger.info(f"Project loaded: {filepath}")
        except Exception as e: logger.error(f"Failed to load project from {filepath}: {e}", exc_info=True); QMessageBox.critical(self, "Load Project Error", f"Could not load project file:\n{filepath}\n\nError: {e}"); self.project_file_path = None; self._update_window_title()
        finally: QApplication.restoreOverrideCursor()
    def _confirm_unsaved_changes(self) -> bool:
        if not self.is_project_modified: return True
        reply = QMessageBox.question(self, "Unsaved Changes", "The current project has unsaved changes. Do you want to save them?", QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel, QMessageBox.StandardButton.Cancel)
        if reply == QMessageBox.StandardButton.Save: return self._handle_save_project_action()
        return reply != QMessageBox.StandardButton.Cancel
    def _handle_save_project_action(self) -> bool:
        if self.project_file_path: return self._perform_project_save(self.project_file_path)
        else: return self._handle_save_project_as_action()
    def _handle_save_project_as_action(self) -> bool:
        last_dir = str(self.project_file_path.parent) if self.project_file_path else self.app_settings.value("Paths/last_project_dir", str(Path.home())); filepath_str, _ = QFileDialog.getSaveFileName(self, "Save Project As", last_dir, "CO2 EOR Project Files (*.tphd);;All Files (*)")
        if not filepath_str: return False
        new_filepath = Path(filepath_str).with_suffix(".tphd")
        if self._perform_project_save(new_filepath): self.project_file_path = new_filepath; self.app_settings.setValue("Paths/last_project_dir", str(new_filepath.parent)); self._update_window_title(); return True
        return False
    def _perform_project_save(self, filepath: Path) -> bool:
        self.show_status_message(f"Saving project to {filepath.name}...", 0); QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            data_to_save = {"application_version": QApplication.applicationVersion(), "project_name": filepath.stem, "well_data_list": self.current_well_data, "reservoir_data": self.current_reservoir_data, "pvt_properties": self.current_pvt_properties, "economic_parameters": self.current_economic_params, "eor_parameters": self.current_eor_params, "operational_parameters": self.current_operational_params, "profile_parameters": self.current_profile_params, "ga_parameters": self.current_ga_params}
            if save_project_to_tphd(data_to_save, filepath): self.set_project_modified(False); self.show_status_message(f"Project saved: {filepath.name}", 5000); logger.info(f"Project saved to {filepath}"); return True
            raise IOError("Project saving function returned failure.")
        except Exception as e: logger.error(f"Failed to save project to {filepath}: {e}", exc_info=True); QMessageBox.critical(self, "Save Project Error", f"Could not save project file:\n{filepath}\n\nError: {e}"); return False
        finally: QApplication.restoreOverrideCursor()
    def _handle_close_project_action(self):
        if self._confirm_unsaved_changes(): logger.info("Closing current project."); self.project_file_path = None; self._initialize_project_data_and_configs(); self.data_management_tab.clear_all_project_data(); self.config_tab.reset_all_to_defaults(); self.mmp_well_analysis_tab.update_data([], None); self.optimization_tab.update_engine(None); self.sensitivity_analyzer_instance = None; self.sensitivity_tab.update_analyzer(None);
        if self.overview_page: self.overview_page.setVisible(True);
        self.main_tab_widget.setVisible(False); self.save_project_action.setEnabled(False); self.save_project_as_action.setEnabled(False); self.close_project_action.setEnabled(False); self._update_window_title(); self.show_status_message("Project closed.", 5000)
    def _handle_quick_start_action(self): pass
    def _show_preferences_dialog(self): QMessageBox.information(self, "Preferences", "Not yet implemented.")
    def _show_report_config_dialog(self): QMessageBox.information(self, "Generate Report", "Not yet implemented.")
    def _show_about_dialog(self): QMessageBox.about(self, f"About {QApplication.applicationName()}", f"<h2>{QApplication.applicationName()}</h2><p>Version: {QApplication.applicationVersion()}</p><p>A CO₂ Enhanced Oil Recovery Suite.</p>")

    # --- FIX: The main fix is applied in this slot ---
    def _on_project_data_model_updated(self, project_data_dict: Dict[str, Any]):
        """Receives finalized data, updates main state, and re-initializes engines."""
        logger.info("MainWindow: Received project_data_updated signal.")
        self.set_project_modified(True)

        # Correctly unpack the payload from the signal into the MainWindow's attributes
        self.current_well_data = project_data_dict.get("well_data_list", [])
        self.current_reservoir_data = project_data_dict.get("reservoir_data")
        self.current_pvt_properties = project_data_dict.get("pvt_properties")
        
        self._reinitialize_engines_and_analysis_tabs()
        self.show_status_message("Core project data models updated.", 3000)

    def _on_app_configuration_changed(self, cat_key: str, param: str, val: Any):
        logger.debug(f"MainWindow: Config changed: {cat_key}.{param} = {val}"); self.set_project_modified(True)
        if self.optimisation_engine_instance:
            if cat_key == EconomicParameters.__name__: self.optimisation_engine_instance.economic_params = self.current_economic_params
            elif cat_key == EORParameters.__name__: self.optimisation_engine_instance.eor_params = self.current_eor_params
            elif cat_key == OperationalParameters.__name__: self.optimisation_engine_instance.operational_params = self.current_operational_params
            elif cat_key == ProfileParameters.__name__: self.optimisation_engine_instance.profile_params = self.current_profile_params
            elif cat_key == GeneticAlgorithmParams.__name__: self.optimisation_engine_instance.ga_params_default_config = self.current_ga_params

    def _on_save_standalone_config_requested(self, config_data: Dict[str, Any]): logger.info(f"ConfigWidget saved standalone configuration.")

    def _reinitialize_engines_and_analysis_tabs(self):
        """Creates new engine instances using the current main window data attributes."""
        logger.info("Re-initializing engines and analysis tabs.")
        self.optimisation_engine_instance = None
        self.sensitivity_analyzer_instance = None
        
        # This check will now pass because the attributes were set in the slot
        if self.current_reservoir_data and self.current_pvt_properties:
            try:
                self.optimisation_engine_instance = OptimizationEngine(
                    reservoir=self.current_reservoir_data, 
                    pvt=self.current_pvt_properties, 
                    eor_params_instance=self.current_eor_params, 
                    ga_params_instance=self.current_ga_params, 
                    economic_params_instance=self.current_economic_params, 
                    operational_params_instance=self.current_operational_params, 
                    profile_params_instance=self.current_profile_params,
                    well_analysis=WellAnalysis(self.current_well_data[0], self.current_pvt_properties) if self.current_well_data else None
                )
                logger.info("OptimizationEngine instance re-created successfully.")
                self.sensitivity_analyzer_instance = SensitivityAnalyzer(engine=self.optimisation_engine_instance)
                logger.info("SensitivityAnalyzer instance re-created successfully.")
            except Exception as e:
                logger.error(f"Failed to create engine/analyzer: {e}", exc_info=True)
                QMessageBox.critical(self, "Engine Error", f"Could not initialize core engines: {e}")
        else:
            logger.warning("Cannot create engines: Missing reservoir or PVT data.")

        # Update all dependent tabs with the new engine instances (or None)
        self.optimization_tab.update_engine(self.optimisation_engine_instance)
        self.sensitivity_tab.update_analyzer(self.sensitivity_analyzer_instance)
        self.mmp_well_analysis_tab.update_data(self.current_well_data, self.current_pvt_properties)

    def _on_optimization_run_completed(self, results: Dict[str, Any]):
        logger.info("MainWindow: Optimization run completed."); self.set_project_modified(True); self.show_status_message("Optimization complete.", 5000)
        if self.optimisation_engine_instance and self.optimisation_engine_instance.results:
            try:
                self.sensitivity_analyzer_instance = SensitivityAnalyzer(engine=self.optimisation_engine_instance, base_case_opt_results=results)
                self.sensitivity_tab.update_analyzer(self.sensitivity_analyzer_instance); logger.info("SensitivityAnalyzer updated with new optimization results as base.")
            except Exception as e: logger.error(f"Error updating SA analyzer post-optimization: {e}", exc_info=True)

    def _provide_context_to_ai_assistant(self, context_req: str): self.ai_assistant_tab.set_context_data_from_main_app(context_req, "Context not available.")
    def save_window_settings(self): self.app_settings.setValue("MainWindow/geometry", self.saveGeometry()); self.app_settings.setValue("MainWindow/state", self.saveState()); logger.debug("Window settings saved.")
    def load_window_settings(self):
        if geom := self.app_settings.value("MainWindow/geometry"): self.restoreGeometry(geom)
        if state := self.app_settings.value("MainWindow/state"): self.restoreState(state)
        logger.debug("Window settings loaded.")
    def closeEvent(self, event: QCloseEvent):
        if self._confirm_unsaved_changes(): self.save_window_settings(); logger.info("Closing application."); event.accept()
        else: event.ignore()