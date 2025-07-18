import logging
import sys
from pathlib import Path
from typing import Optional, Any, Dict, List
from copy import deepcopy
import traceback

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QStatusBar, QMessageBox,
    QFileDialog, QTabWidget, QApplication, QLabel, QProgressBar,
    QSplitter, QHBoxLayout, QStackedLayout
)
from PyQt6.QtGui import QAction, QIcon, QKeySequence, QCloseEvent
from PyQt6.QtCore import Qt, pyqtSignal, QSettings, QTimer, pyqtSlot

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
    from co2eor_optimizer.config_manager import ConfigManager
    from co2eor_optimizer.utils.project_file_handler import save_project_to_tphd, load_project_from_tphd
    from co2eor_optimizer.utils.units_manager import units_manager
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
    from co2eor_optimizer.help_manager import HelpManager
    from co2eor_optimizer.ui.dialogs.parameter_help_dialog import HelpPanel
    from co2eor_optimizer.core.data_models import (
        WellData, ReservoirData, PVTProperties,
        EconomicParameters, EORParameters, OperationalParameters,
        ProfileParameters, GeneticAlgorithmParams, BayesianOptimizationParams
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
    project_file_path: Optional[Path] = None
    is_project_modified: bool = False
    help_requested = pyqtSignal(str)
    
    current_well_data: List[WellData]
    current_reservoir_data: Optional[ReservoirData]
    current_pvt_properties: Optional[PVTProperties]
    current_mmp_value: Optional[float] = None
    
    current_economic_params: EconomicParameters
    current_eor_params: EORParameters
    current_operational_params: OperationalParameters
    current_profile_params: ProfileParameters
    current_ga_params: GeneticAlgorithmParams
    current_bo_params: BayesianOptimizationParams

    optimisation_engine_instance: Optional[OptimizationEngine] = None
    sensitivity_analyzer_instance: Optional[SensitivityAnalyzer] = None

    def __init__(self, app_settings: QSettings):
        super().__init__(None)
        self.app_settings = app_settings
        
        config_dir_path = str(Path(sys.executable).parent / 'config' if getattr(sys, 'frozen', False) else Path(__file__).parent.parent / 'config')
        self.default_config_loader = ConfigManager(config_dir_path=config_dir_path, require_config=False, autoload=True)
        if not self.default_config_loader.is_loaded:
            QMessageBox.warning(self, "Config Warning", f"Could not load default configurations from {config_dir_path}. Using hardcoded defaults.")

        self._initialize_project_data_and_configs()
        self.report_generator = ReportGenerator(units_manager)
        
        self._setup_ui_structure()
        self._create_actions()
        self._create_menu_bar()
        self._create_status_bar()
        self._create_tool_bar()
        self._create_help_system_connections()
        logger.info("MainWindow UI structure, actions, and menus initialized.")

        self._setup_initial_view()
        self.load_window_settings()
        self._update_window_title()

        self.resource_monitor_timer = QTimer(self)
        self.resource_monitor_timer.setInterval(2000)
        self.resource_monitor_timer.timeout.connect(self._update_resource_monitors)
        self.resource_monitor_timer.start()

    def _initialize_project_data_and_configs(self, from_project_load: bool = False):
        logger.info(f"Initializing project data and configurations (from_project_load={from_project_load}).")
        self.current_well_data = []
        self.current_reservoir_data = None
        self.current_pvt_properties = None
        self.current_mmp_value = None

        if not from_project_load:
            try:
                self.current_economic_params = EconomicParameters.from_config_dict(self.default_config_loader.get_section("EconomicParametersDefaults") or {})
                self.current_eor_params = EORParameters.from_config_dict(self.default_config_loader.get_section("EORParametersDefaults") or {})
                self.current_operational_params = OperationalParameters.from_config_dict(self.default_config_loader.get_section("OperationalParametersDefaults") or {})
                self.current_profile_params = ProfileParameters.from_config_dict(self.default_config_loader.get_section("ProfileParametersDefaults") or {})
                self.current_ga_params = GeneticAlgorithmParams.from_config_dict(self.default_config_loader.get_section("GeneticAlgorithmParamsDefaults") or {})
                self.current_bo_params = BayesianOptimizationParams.from_config_dict(self.default_config_loader.get_section("BayesianOptimizationParamsDefaults") or {})
            except Exception as e:
                logger.critical(f"Failed to initialize one or more configuration dataclasses from defaults. Error: {e}", exc_info=True)
                QMessageBox.critical(self, "Configuration Error", f"Could not load default configuration files from the 'config' directory.\n\n{e}")
                raise
        
        self.optimisation_engine_instance = None
        self.sensitivity_analyzer_instance = None
        self.is_project_modified = False if not from_project_load else self.is_project_modified

    def _setup_ui_structure(self):
        self.setWindowTitle(QApplication.applicationName())
        self.setWindowIcon(QIcon.fromTheme("application-x-executable"))
        
        # This is the main widget that will hold our view-switching layout
        self.main_central_widget = QWidget(self)
        self.setCentralWidget(self.main_central_widget)
        
        # QStackedLayout is the correct tool for switching between full-page views
        self.stacked_layout = QStackedLayout(self.main_central_widget)
        self.stacked_layout.setContentsMargins(0, 0, 0, 0)

        # -- Page 1: The Overview/Welcome Page --
        self.overview_page = OverviewPageWidget(self)
        self.overview_page.start_new_project_requested.connect(self._handle_start_new_project_action)
        self.overview_page.load_existing_project_requested.connect(self._handle_load_project_action)
        self.overview_page.quick_start_requested.connect(self._handle_quick_start_action)
        
        # -- Page 2: The Main Application Interface (Tabs + Help Panel) --
        main_app_widget = QWidget() # This widget will contain the splitter
        main_app_layout = QHBoxLayout(main_app_widget)
        main_app_layout.setContentsMargins(0,0,0,0)

        self.help_panel = HelpPanel(self)
        self.main_splitter = QSplitter(self)
        main_app_layout.addWidget(self.main_splitter)

        self._setup_main_app_tabs_container()
        
        self.main_splitter.addWidget(self.main_tab_widget)
        self.main_splitter.addWidget(self.help_panel)
        self.main_splitter.setCollapsible(0, False)
        self.main_splitter.setCollapsible(1, True)
        self.main_splitter.setStretchFactor(0, 1)
        self.main_splitter.setStretchFactor(1, 0)
        self.help_panel.hide()

        # -- Add both pages to the stacked layout. Index 0 = overview, Index 1 = main app --
        self.stacked_layout.addWidget(self.overview_page)
        self.stacked_layout.addWidget(main_app_widget)

    def _setup_initial_view(self):
        # On startup, show the overview page (index 0)
        self.stacked_layout.setCurrentIndex(0)
        self.save_project_action.setEnabled(False)
        self.save_project_as_action.setEnabled(False)
        self.close_project_action.setEnabled(False)

    def _setup_main_app_tabs_container(self):
        self.main_tab_widget = QTabWidget(self)
        self.main_tab_widget.setTabPosition(QTabWidget.TabPosition.North)
        self.main_tab_widget.setMovable(True)
        
        # All tab creation code is unchanged
        self.config_tab = ConfigWidget(self); self.config_tab.configurations_updated.connect(self._on_app_configurations_updated); self.config_tab.save_configuration_to_file_requested.connect(self._on_save_standalone_config_requested); self.main_tab_widget.addTab(self.config_tab, QIcon.fromTheme("preferences-system"), "1. Configuration")
        self.data_management_tab = DataManagementWidget(self); self.data_management_tab.project_data_updated.connect(self._on_project_data_model_updated); self.data_management_tab.status_message_updated.connect(self.show_status_message); self.main_tab_widget.addTab(self.data_management_tab, QIcon.fromTheme("document-properties"), "2. Data Management")
        self.mmp_well_analysis_tab = MMPWellAnalysisWidget(self); self.mmp_well_analysis_tab.representative_mmp_calculated.connect(self._on_representative_mmp_updated); self.main_tab_widget.addTab(self.mmp_well_analysis_tab, QIcon.fromTheme("view-plot"), "3. MMP & Well Analysis")
        self.optimization_tab = OptimizationWidget(self); self.optimization_tab.optimization_completed.connect(self._on_optimization_run_completed); self.optimization_tab.open_configuration_requested.connect(self._open_configuration_tab); self.config_tab.configurations_updated.connect(self.optimization_tab.on_configurations_updated); self.main_tab_widget.addTab(self.optimization_tab, QIcon.fromTheme("system-run"), "4. Optimization")
        self.sensitivity_tab = SensitivityWidget(self); self.main_tab_widget.addTab(self.sensitivity_tab, QIcon.fromTheme("view-statistics"), "5. Sensitivity Analysis")
        self.uq_tab = UQWidget(self); self.main_tab_widget.addTab(self.uq_tab, QIcon.fromTheme("view-process-users"), "6. Uncertainty Quantification")
        self.ai_assistant_tab = AIAssistantWidget(self.app_settings, self); self.ai_assistant_tab.request_context_data.connect(self._provide_context_to_ai_assistant); self.main_tab_widget.addTab(self.ai_assistant_tab, QIcon.fromTheme("preferences-desktop-ai"), "7. AI Assistant")
        self.logging_tab = LoggingWidget(self); self.main_tab_widget.addTab(self.logging_tab, QIcon.fromTheme("text-x-generic"), "8. Application Logs")
        
    def _transition_to_main_app_view(self, focus_tab_index: int = 0):
        # Switch the stacked layout to show the main app (index 1)
        self.stacked_layout.setCurrentIndex(1)
        # Set a sensible default for the splitter sizes
        self.main_splitter.setSizes([self.width() - 350, 350])
        
        self.main_tab_widget.setCurrentIndex(focus_tab_index)
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
        menu_bar = self.menuBar(); file_menu = menu_bar.addMenu("&File"); file_menu.addAction(self.new_project_action); file_menu.addAction(self.open_project_action); file_menu.addSeparator(); file_menu.addAction(self.save_project_action); file_menu.addAction(self.save_project_as_action); file_menu.addSeparator(); file_menu.addAction(self.close_project_action); file_menu.addSeparator(); file_menu.addAction(self.exit_action); edit_menu = menu_bar.addMenu("&Edit"); edit_menu.addAction(self.settings_action); tools_menu = menu_bar.addMenu("&Tools"); tools_menu.addAction(self.generate_report_action); 
        help_menu = menu_bar.addMenu("&Help")
        self.show_help_action = QAction(QIcon.fromTheme("help-contextual"), "Show Help Panel", self)
        self.show_help_action.triggered.connect(lambda: self.request_help("Global.overview"))
        help_menu.addAction(self.show_help_action)
        help_menu.addSeparator()
        help_menu.addAction(self.about_action)
        help_menu.addAction(self.about_qt_action)

    def _create_status_bar(self):
        self.status_bar = QStatusBar(self); self.setStatusBar(self.status_bar)
        if PSUTIL_AVAILABLE: self.cpu_label = QLabel(" CPU: --% "); self.status_bar.addPermanentWidget(self.cpu_label); self.ram_label = QLabel(" RAM: --% "); self.status_bar.addPermanentWidget(self.ram_label)
        if GPUTIL_AVAILABLE and GPUtil.getGPUs(): self.gpu_label = QLabel(" GPU: --% "); self.status_bar.addPermanentWidget(self.gpu_label)
        self.status_progress_bar = QProgressBar(self); self.status_progress_bar.setRange(0,100); self.status_progress_bar.setVisible(False); self.status_progress_bar.setFixedSize(150, 16); self.status_bar.addPermanentWidget(self.status_progress_bar)
        self.show_status_message("Ready.", 5000)

    def _create_help_system_connections(self):
        self.help_requested.connect(self.help_panel.show_help_for)
        self.help_panel.closed.connect(self._on_help_panel_closed)
        
    @pyqtSlot(str)
    def request_help(self, key: str):
        if self.main_splitter.sizes()[1] == 0:
            self.main_splitter.setSizes([self.width() - 350, 350])
        self.help_requested.emit(key)
        
    @pyqtSlot()
    def _on_help_panel_closed(self):
        sizes = self.main_splitter.sizes()
        if sizes[1] > 0:
            self.main_splitter.setSizes([sum(sizes), 0])
            
    def _update_resource_monitors(self):
        if PSUTIL_AVAILABLE: self.cpu_label.setText(f" CPU: {psutil.cpu_percent():.0f}% "); self.ram_label.setText(f" RAM: {psutil.virtual_memory().percent:.0f}% ")
        if GPUTIL_AVAILABLE and hasattr(self, 'gpu_label'):
            try: gpu = GPUtil.getGPUs()[0]; self.gpu_label.setText(f" GPU: {gpu.load*100:.0f}% ")
            except Exception: self.gpu_label.setText(" GPU: N/A ")
                
    def _create_tool_bar(self): pass
    def _update_window_title(self): title = QApplication.applicationName(); title += f" - {self.project_file_path.name}" if self.project_file_path else " - New Project"; self.setWindowTitle(title + "*" if self.is_project_modified else title)
    def set_project_modified(self, modified: bool = True):
        if self.is_project_modified != modified: self.is_project_modified = modified; self._update_window_title()
    def show_status_message(self, message: str, timeout: int = 3000): self.status_bar.showMessage(message, timeout); logger.debug(f"Status: {message}")
    
    def _handle_start_new_project_action(self):
        if self._confirm_unsaved_changes():
            logger.info("Starting new project.")
            self.project_file_path = None
            self._initialize_project_data_and_configs()
            all_configs = {EconomicParameters.__name__: deepcopy(self.current_economic_params), EORParameters.__name__: deepcopy(self.current_eor_params), OperationalParameters.__name__: deepcopy(self.current_operational_params), ProfileParameters.__name__: deepcopy(self.current_profile_params), GeneticAlgorithmParams.__name__: deepcopy(self.current_ga_params), BayesianOptimizationParams.__name__: deepcopy(self.current_bo_params),}
            self.config_tab.update_configurations(all_configs)
            self.data_management_tab.clear_all_project_data()
            self.mmp_well_analysis_tab.update_data([], None)
            self.optimization_tab.update_engine(None)
            self.sensitivity_tab.update_analyzer(None)
            self._update_window_title()
            self.set_project_modified(False)
            self._transition_to_main_app_view(focus_tab_index=1)
            self.show_status_message("New project started. Configure parameters and load data.", 5000)

    def _handle_load_project_action(self):
        if not self._confirm_unsaved_changes(): return
        last_dir = self.app_settings.value("Paths/last_project_dir", str(Path.home())); filepath_str, _ = QFileDialog.getOpenFileName(self, "Open Project File", last_dir, "CO2 EOR Project Files (*.tphd);;All Files (*)")
        if not filepath_str: return
        filepath = Path(filepath_str); self.app_settings.setValue("Paths/last_project_dir", str(filepath.parent)); self.show_status_message(f"Loading project: {filepath.name}...", 0); QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor); QTimer.singleShot(50, lambda: self._perform_project_load(filepath))
    
    def _perform_project_load(self, filepath: Path):
        try:
            project_data_dict = load_project_from_tphd(filepath)
            if project_data_dict is None: raise IOError("Failed to load or parse project file.")
            self._initialize_project_data_and_configs(from_project_load=True)
            self.current_well_data = project_data_dict.get("well_data_list", [])
            self.current_reservoir_data = project_data_dict.get("reservoir_data")
            self.current_pvt_properties = project_data_dict.get("pvt_properties")
            self.current_mmp_value = project_data_dict.get("mmp_value")
            self.current_economic_params = project_data_dict.get("economic_parameters", EconomicParameters())
            self.current_eor_params = project_data_dict.get("eor_parameters", EORParameters())
            self.current_operational_params = project_data_dict.get("operational_parameters", OperationalParameters())
            self.current_profile_params = project_data_dict.get("profile_parameters", ProfileParameters())
            self.current_ga_params = project_data_dict.get("ga_parameters", GeneticAlgorithmParams())
            self.current_bo_params = project_data_dict.get("bo_parameters", BayesianOptimizationParams())
            loaded_configs = {EconomicParameters.__name__: deepcopy(self.current_economic_params), EORParameters.__name__: deepcopy(self.current_eor_params), OperationalParameters.__name__: deepcopy(self.current_operational_params), ProfileParameters.__name__: deepcopy(self.current_profile_params), GeneticAlgorithmParams.__name__: deepcopy(self.current_ga_params), BayesianOptimizationParams.__name__: deepcopy(self.current_bo_params),}
            self.config_tab.update_configurations(loaded_configs)
            self.data_management_tab._populate_review_ui()
            self._reinitialize_engines_and_analysis_tabs()
            self.project_file_path = filepath
            self.set_project_modified(False)
            self._update_window_title()
            self._transition_to_main_app_view(focus_tab_index=0)
            self.show_status_message(f"Project '{filepath.name}' loaded successfully.", 5000)
            logger.info(f"Project loaded: {filepath}")
        except Exception as e:
            logger.error(f"Failed to load project from {filepath}: {e}", exc_info=True)
            QMessageBox.critical(self, "Load Project Error", f"Could not load project file:\n{filepath}\n\nError: {e}")
            self.project_file_path = None; self._update_window_title()
        finally:
            QApplication.restoreOverrideCursor()

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
            data_to_save = {"application_version": QApplication.applicationVersion(), "project_name": filepath.stem, "well_data_list": self.current_well_data, "reservoir_data": self.current_reservoir_data, "pvt_properties": self.current_pvt_properties, "mmp_value": self.current_mmp_value, "economic_parameters": self.current_economic_params, "eor_parameters": self.current_eor_params, "operational_parameters": self.current_operational_params, "profile_parameters": self.current_profile_params, "ga_parameters": self.current_ga_params, "bo_parameters": self.current_bo_params,}
            if save_project_to_tphd(data_to_save, filepath):
                self.set_project_modified(False); self.show_status_message(f"Project saved: {filepath.name}", 5000); logger.info(f"Project saved to {filepath}"); return True
            raise IOError("Project saving function returned failure.")
        except Exception as e:
            logger.error(f"Failed to save project to {filepath}: {e}", exc_info=True); QMessageBox.critical(self, "Save Project Error", f"Could not save project file:\n{filepath}\n\nError: {e}"); return False
        finally:
            QApplication.restoreOverrideCursor()

    def _handle_close_project_action(self):
        if self._confirm_unsaved_changes():
            logger.info("Closing current project.")
            self.project_file_path = None
            self._initialize_project_data_and_configs()
            all_configs = {EconomicParameters.__name__: self.current_economic_params, EORParameters.__name__: self.current_eor_params, OperationalParameters.__name__: self.current_operational_params, ProfileParameters.__name__: self.current_profile_params, GeneticAlgorithmParams.__name__: self.current_ga_params, BayesianOptimizationParams.__name__: self.current_bo_params}
            self.config_tab.update_configurations(all_configs)
            self.data_management_tab.clear_all_project_data()
            self.mmp_well_analysis_tab.update_data([], None)
            self.optimization_tab.update_engine(None)
            self.sensitivity_tab.update_analyzer(None)
            self.stacked_layout.setCurrentIndex(0)
            self.save_project_action.setEnabled(False)
            self.save_project_as_action.setEnabled(False)
            self.close_project_action.setEnabled(False)
            self._update_window_title()
            self.show_status_message("Project closed.", 5000)
            
    def _handle_quick_start_action(self): pass
    def _show_preferences_dialog(self): QMessageBox.information(self, "Preferences", "Not yet implemented.")
    def _show_report_config_dialog(self): QMessageBox.information(self, "Generate Report", "Not yet implemented.")
    def _show_about_dialog(self): QMessageBox.about(self, f"About {QApplication.applicationName()}", f"<h2>{QApplication.applicationName()}</h2><p>Version: {QApplication.applicationVersion()}</p><p>A CO₂ Enhanced Oil Recovery Suite.</p>")

    @pyqtSlot(dict)
    def _on_project_data_model_updated(self, project_data_dict: Dict[str, Any]):
        logger.info("MainWindow: Received project_data_updated signal."); self.set_project_modified(True); self.current_well_data = project_data_dict.get("well_data_list", []); self.current_reservoir_data = project_data_dict.get("reservoir_data"); self.current_pvt_properties = project_data_dict.get("pvt_properties"); is_data_finalization = project_data_dict.get("is_data_finalization", False); self._reinitialize_engines_and_analysis_tabs(skip_calculations=is_data_finalization); self.show_status_message("Core project data models updated.", 3000)

    @pyqtSlot(dict)
    def _on_app_configurations_updated(self, new_configs: Dict[str, Any]):
        logger.info("MainWindow state updating from ConfigWidget."); self.set_project_modified(True); self.current_economic_params = new_configs.get(EconomicParameters.__name__, self.current_economic_params); self.current_eor_params = new_configs.get(EORParameters.__name__, self.current_eor_params); self.current_operational_params = new_configs.get(OperationalParameters.__name__, self.current_operational_params); self.current_profile_params = new_configs.get(ProfileParameters.__name__, self.current_profile_params); self.current_ga_params = new_configs.get(GeneticAlgorithmParams.__name__, self.current_ga_params); self.current_bo_params = new_configs.get(BayesianOptimizationParams.__name__, self.current_bo_params); self._reinitialize_engines_and_analysis_tabs()

    @pyqtSlot(float)
    def _on_representative_mmp_updated(self, mmp_value: float):
        logger.info(f"MainWindow received representative MMP value: {mmp_value:.2f} psi")
        if self.current_mmp_value != mmp_value: self.current_mmp_value = mmp_value; self.set_project_modified(True); self._reinitialize_engines_and_analysis_tabs(); self.show_status_message(f"Optimization Engine updated with MMP = {mmp_value:.1f} psi", 5000)

    @pyqtSlot(dict)
    def _on_save_standalone_config_requested(self, config_data: Dict[str, Any]):
        logger.info(f"ConfigWidget saved standalone configuration.")

    def _reinitialize_engines_and_analysis_tabs(self, skip_calculations: bool = False):
        logger.info(f"Re-initializing engines and analysis tabs (skip_calculations={skip_calculations})."); self.optimisation_engine_instance = None; self.sensitivity_analyzer_instance = None
        if self.current_reservoir_data and self.current_pvt_properties:
            try:
                self.optimisation_engine_instance = OptimizationEngine(reservoir=deepcopy(self.current_reservoir_data), pvt=deepcopy(self.current_pvt_properties), eor_params_instance=deepcopy(self.current_eor_params), ga_params_instance=deepcopy(self.current_ga_params), bo_params_instance=deepcopy(self.current_bo_params), economic_params_instance=deepcopy(self.current_economic_params), operational_params_instance=deepcopy(self.current_operational_params), profile_params_instance=deepcopy(self.current_profile_params), well_data_list=self.current_well_data, mmp_init_override=self.current_mmp_value, skip_auto_calculations=skip_calculations)
                logger.info("OptimizationEngine instance created.")
                if not skip_calculations and self.current_mmp_value is None: logger.info("No MMP override present. Triggering self-calculation for the Optimization Engine."); self.optimisation_engine_instance.calculate_mmp()
                self.sensitivity_analyzer_instance = SensitivityAnalyzer(engine=self.optimisation_engine_instance); logger.info("SensitivityAnalyzer instance re-created successfully.")
            except Exception as e: logger.error(f"Failed to create engine/analyzer: {e}", exc_info=True); QMessageBox.critical(self, "Engine Error", f"Could not initialize core engines: {e}")
        else: logger.warning("Cannot create engines: Missing reservoir or PVT data.")
        self.optimization_tab.update_engine(self.optimisation_engine_instance); self.sensitivity_tab.update_analyzer(self.sensitivity_analyzer_instance); self.mmp_well_analysis_tab.update_data(self.current_well_data, self.current_pvt_properties)

    @pyqtSlot()
    def _open_configuration_tab(self):
        self.main_tab_widget.setCurrentIndex(0); self.show_status_message("Configuration panel opened.", 3000)
        
    @pyqtSlot(dict)
    def _on_optimization_run_completed(self, results: Dict[str, Any]):
        logger.info("MainWindow: Optimization run completed."); self.set_project_modified(True); self.show_status_message("Optimization complete.", 5000)
        if self.optimisation_engine_instance and self.optimisation_engine_instance.results:
            try: self.sensitivity_analyzer_instance = SensitivityAnalyzer(engine=self.optimisation_engine_instance); self.sensitivity_tab.update_analyzer(self.sensitivity_analyzer_instance); logger.info("SensitivityAnalyzer updated with new optimization results as base.")
            except Exception as e: logger.error(f"Error updating Sensitivity Analyzer post-optimization: {e}", exc_info=True)

    @pyqtSlot(str)
    def _provide_context_to_ai_assistant(self, context_req: str):
        context_data = {};
        if self.optimisation_engine_instance: context_data['optimization_engine_state'] = self.optimisation_engine_instance.results
        if self.current_reservoir_data: context_data['reservoir_summary'] = {'grid_dims': self.current_reservoir_data.runspec.get('DIMENSIONS') if self.current_reservoir_data.runspec else 'N/A', 'ooip': self.current_reservoir_data.ooip_stb}
        self.ai_assistant_tab.set_context_data_from_main_app(context_req, context_data)

    def save_window_settings(self):
        self.app_settings.setValue("MainWindow/geometry", self.saveGeometry())
        self.app_settings.setValue("MainWindow/state", self.saveState())
        self.app_settings.setValue("MainWindow/splitterSizes", self.main_splitter.saveState())
        logger.debug("Window settings saved.")

    def load_window_settings(self):
        if geom := self.app_settings.value("MainWindow/geometry"): self.restoreGeometry(geom)
        if state := self.app_settings.value("MainWindow/state"): self.restoreState(state)
        if splitter_state := self.app_settings.value("MainWindow/splitterSizes"):
            self.main_splitter.restoreState(splitter_state)
        else:
            self.main_splitter.setSizes([self.width(), 0])
        
        if self.stacked_layout.currentIndex() == 0:
            self.help_panel.hide()

        logger.debug("Window settings loaded.")

    def closeEvent(self, event: QCloseEvent):
        if self._confirm_unsaved_changes():
            self.save_window_settings()
            logger.info("Closing application.")
            event.accept()
        else:
            event.ignore()