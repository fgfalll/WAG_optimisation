import logging
import sys
from pathlib import Path
from typing import Optional, Any, Dict, List
from copy import deepcopy
import traceback
from dataclasses import asdict
import numpy as np

from path_utils import get_config_dir, get_ui_assets_dir

import pandas as pd
import base64
import io
from plotly.io import to_image
import plotly.graph_objects as go


import qtawesome as qta
from PyQt6.QtWidgets import (
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QStatusBar,
    QMessageBox,
    QFileDialog,
    QTabWidget,
    QApplication,
    QLabel,
    QProgressBar,
    QSplitter,
    QHBoxLayout,
    QStackedLayout,
    QDialog,
    QProgressDialog,
    QGroupBox,
    QTextBrowser,
)
from PyQt6.QtGui import QAction, QIcon, QKeySequence, QCloseEvent, QFont
from PyQt6.QtCore import Qt, pyqtSignal, QSettings, QTimer, pyqtSlot, QEvent
from PyQt6.QtGui import QPalette, QColor

try:
    from PyQt6.QtWebEngineWidgets import QWebEngineView
except ImportError:
    QWebEngineView = None

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

logger = logging.getLogger(__name__)

try:
    from config_manager import ConfigManager
    from utils.project_file_handler import save_project_to_tphd, load_project_from_tphd
    from utils.units_manager import units_manager
    from utils.report_generator import ReportGenerator
    from ui.overview_page import OverviewPageWidget
    from ui.data_management_widget import DataManagementWidget
    from ui.config_widget import ConfigWidget
    from ui.optimization_widget import OptimizationWidget
    from ui.analysis_widget import AnalysisWidget
    from ui.ai_assistant_widget import AIAssistantWidget
    from ui.dialogs.report_config_dialog import ReportConfigDialog
    from ui.dialogs.scientific_justification_dialog import ScientificJustificationWidget
    from help_manager import HelpManager
    from ui.dialogs.parameter_help_dialog import HelpPanel
    from core.data_models import (
        WellData,
        ReservoirData,
        PVTProperties,
        EconomicParameters,
        EORParameters,
        OperationalParameters,
        ProfileParameters,
        GeneticAlgorithmParams,
        BayesianOptimizationParams,
        AdvancedEngineParams,
    )
    from core.optimisation_engine import OptimizationEngine
    from analysis.sensitivity_analyzer import SensitivityAnalyzer
    from analysis.uq_engine import UncertaintyQuantificationEngine
    from analysis.well_analysis import WellAnalysis
    from utils.file_association import FileAssociationManager
    from utils.preferences_manager import get_preferences_manager
    from ui.dialogs.preferences_dialog import PreferencesDialog
    from ui.workers.ai_query_worker import AIQueryWorker

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
        logging.critical(log_msg, exc_info=True)
    sys.exit(1)


class MainWindow(QMainWindow):
    project_file_path: Optional[Path] = None
    is_project_modified: bool = False
    help_requested = pyqtSignal(str)
    progress_updated = pyqtSignal(int, str)
    show_message = pyqtSignal(str, str, str)

    current_well_data: List[WellData]
    current_reservoir_data: Optional[ReservoirData]
    current_pvt_properties: Optional[PVTProperties]
    current_mmp_value: Optional[float] = None
    current_project_name: Optional[str] = None

    current_economic_params: EconomicParameters
    current_eor_params: EORParameters
    current_operational_params: OperationalParameters
    current_profile_params: ProfileParameters
    current_ga_params: GeneticAlgorithmParams
    current_bo_params: BayesianOptimizationParams

    optimisation_engine_instance: Optional[OptimizationEngine] = None
    sensitivity_analyzer_instance: Optional[SensitivityAnalyzer] = None
    progress_dialog: Optional[QProgressDialog] = None

    def __init__(
        self,
        app_settings: QSettings,
        preferences_manager=None,
        startup_action: str = "show_overview",
        qt_log_handler: Optional[Any] = None,
    ):
        super().__init__(None)
        self.app_settings = app_settings
        self.preferences_manager = preferences_manager
        self.startup_action = startup_action
        self.auto_save_timer = None
        self.qt_log_handler = qt_log_handler

        config_dir_path = str(get_config_dir())
        self.default_config_loader = ConfigManager(
            config_dir_path=config_dir_path, require_config=False, autoload=True
        )
        if not self.default_config_loader.is_loaded:
            QMessageBox.warning(
                self,
                self.tr("Config Warning"),
                self.tr(
                    f"Could not load default configurations from {config_dir_path}. Using hardcoded defaults."
                ),
            )

        self._initialize_project_data_and_configs()
        self.report_generator = ReportGenerator(units_manager)

        self._setup_ui_structure()
        self._create_actions()
        self._create_menu_bar()
        self._create_status_bar()
        self._create_tool_bar()
        self._create_help_system_connections()
        self.progress_updated.connect(self._update_report_progress)
        self.show_message.connect(self._show_message_box)

        self.retranslateUi()
        logger.info("MainWindow UI structure, actions, and menus initialized.")

        self._setup_initial_view()
        self.load_window_settings()
        self._update_window_title()

        # Connect preferences change signals
        if self.preferences_manager:
            self.preferences_manager.general_preferences_changed.connect(
                self._on_general_preferences_changed
            )
            self.preferences_manager.display_preferences_changed.connect(
                self._on_display_preferences_changed
            )
            self.preferences_manager.units_preferences_changed.connect(
                self._on_units_preferences_changed
            )
            self.preferences_manager.language_preferences_changed.connect(
                self._on_language_preferences_changed
            )
            self.preferences_manager.advanced_preferences_changed.connect(
                self._on_advanced_preferences_changed
            )
            self.preferences_manager.preferences_reset.connect(self._on_preferences_reset)

            # Setup auto-save timer if enabled
            self._setup_auto_save_timer()

        self.resource_monitor_timer = QTimer(self)
        self.resource_monitor_timer.setInterval(2000)
        self.resource_monitor_timer.timeout.connect(self._update_resource_monitors)
        self.resource_monitor_timer.start()

    def _generate_ai_summaries(self, ai_features: Dict[str, bool], report_data: Dict[str, Any]):
        from PyQt6.QtCore import QEventLoop

        def run_sync_query(prompt: str) -> str:
            ai_prefs = self.ai_assistant_tab.pref_manager.ai
            service = ai_prefs.active_service
            service_config = ai_prefs.services.get(service, {})
            api_key = service_config.get("api_key")
            base_url = service_config.get("base_url")
            model_name = self.ai_assistant_tab.model_combo.currentText()  # Use current model

            if not api_key:
                raise ValueError(f"API key for {service} is not configured.")

            loop = QEventLoop()
            worker = AIQueryWorker(api_key, model_name, prompt, service, base_url)
            result = [""]
            error = [""]

            def on_finished(response):
                result[0] = response
                loop.quit()

            def on_error(err_msg):
                error[0] = err_msg
                loop.quit()

            worker.finished_result_ready.connect(on_finished)
            worker.error_occurred.connect(on_error)
            worker.start()
            loop.exec()

            if error[0]:
                raise Exception(f"AI query failed: {error[0]}")
            return result[0]

        if ai_features.get("optimization_results") and report_data.get("optimization_results"):
            prompt = f"Summarize the following optimization results for a technical report: {report_data['optimization_results']}"
            summary = run_sync_query(prompt)
            report_data["ai_summary_optimization_results"] = summary

        # Add other sections here

    def _initialize_project_data_and_configs(self, from_project_load: bool = False):
        logger.info(
            f"Initializing project data and configurations (from_project_load={from_project_load})."
        )

        if not from_project_load:
            # Only reset data when not loading from project
            self.current_well_data = []
            self.current_reservoir_data = None
            self.current_pvt_properties = None
            self.current_mmp_value = None
            self.current_project_name = None

            try:
                self.current_economic_params = EconomicParameters.from_config_dict(
                    self.default_config_loader.get_section("EconomicParametersDefaults") or {}
                )
                self.current_eor_params = EORParameters.from_config_dict(
                    self.default_config_loader.get_section("EORParametersDefaults") or {}
                )
                self.current_operational_params = OperationalParameters.from_config_dict(
                    self.default_config_loader.get_section("OperationalParametersDefaults") or {}
                )
                self.current_profile_params = ProfileParameters.from_config_dict(
                    self.default_config_loader.get_section("ProfileParametersDefaults") or {}
                )
                self.current_ga_params = GeneticAlgorithmParams.from_config_dict(
                    self.default_config_loader.get_section("GeneticAlgorithmParamsDefaults") or {}
                )
                self.current_bo_params = BayesianOptimizationParams.from_config_dict(
                    self.default_config_loader.get_section("BayesianOptimizationParamsDefaults")
                    or {}
                )
                self.current_pvt_properties = PVTProperties.from_config_dict(
                    self.default_config_loader.get_section("PVTPropertiesDefaults") or {}
                )
                # Initialize advanced engine params (includes engine selection)
                self.current_advanced_engine_params = AdvancedEngineParams.from_config_dict(
                    self.default_config_loader.get_section("AdvancedEngineParamsDefaults") or {}
                )
                logger.info("Initialized PVT properties with default viscosities from config.")

            except Exception as e:
                logger.critical(
                    f"Failed to initialize one or more configuration dataclasses from defaults. Error: {e}",
                    exc_info=True,
                )
                QMessageBox.critical(
                    self,
                    self.tr("Configuration Error"),
                    self.tr(
                        "Could not load default configuration files from the 'config' directory.\n\n{e}"
                    ),
                )
                raise

        self.optimisation_engine_instance = None
        self.sensitivity_analyzer_instance = None
        self.is_project_modified = False if not from_project_load else self.is_project_modified

    def _ensure_dataclass_instance(self, data, dataclass_type):
        if isinstance(data, dataclass_type):
            return data
        elif isinstance(data, dict):
            try:
                return dataclass_type.from_config_dict(data)
            except Exception as e:
                logger.error(f"Failed to create {dataclass_type.__name__} from dict: {e}")
                return dataclass_type()
        else:
            return dataclass_type()

    def _setup_ui_structure(self):
        icon_path = get_ui_assets_dir() / "main_ico.ico"
        if icon_path.exists():
            self.setWindowIcon(QIcon(str(icon_path)))
        else:
            logger.warning(f"Custom icon not found at {icon_path}, using default theme icon")
            self.setWindowIcon(qta.icon("fa5s.cogs"))

        # Set reasonable minimum size based on screen constraints
        self._set_adaptive_minimum_size()

        self.main_central_widget = QWidget(self)
        self.setCentralWidget(self.main_central_widget)

        self.stacked_layout = QStackedLayout(self.main_central_widget)
        self.stacked_layout.setContentsMargins(0, 0, 0, 0)

        self.overview_page = OverviewPageWidget(self, preferences_manager=self.preferences_manager)
        self.overview_page.start_new_project_requested.connect(
            self._handle_start_new_project_action
        )
        self.overview_page.load_existing_project_requested.connect(self._handle_load_project_action)
        self.overview_page.quick_start_requested.connect(self._handle_quick_start_action)
        self.overview_page.open_recent_project_requested.connect(self._handle_open_recent_project)

        main_app_widget = QWidget()
        main_app_layout = QHBoxLayout(main_app_widget)
        main_app_layout.setContentsMargins(0, 0, 0, 0)

        self.help_panel = HelpPanel(self)
        self.help_panel.setMinimumWidth(250)
        self.help_panel.setMaximumWidth(500)
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

        self.stacked_layout.addWidget(self.overview_page)
        self.stacked_layout.addWidget(main_app_widget)

    def _set_adaptive_minimum_size(self):
        """Set minimum window size based on screen constraints and content requirements."""
        screen_geometry = QApplication.primaryScreen().availableGeometry()

        # Calculate minimum size that respects screen constraints
        min_width = 1200  # Reasonable minimum width for the application
        min_height = 800  # Reasonable minimum height for content

        # Ensure minimum size doesn't exceed screen size
        min_width = min(min_width, screen_geometry.width() - 100)  # Leave some margin
        min_height = min(min_height, screen_geometry.height() - 100)

        # Set the minimum size
        self.setMinimumSize(min_width, min_height)

        logger.debug(
            f"Set adaptive minimum size: {min_width}x{min_height} "
            f"(Screen: {screen_geometry.width()}x{screen_geometry.height()})"
        )

    def _setup_initial_view(self):
        self.save_project_action.setEnabled(False)
        self.save_project_as_action.setEnabled(False)
        self.close_project_action.setEnabled(False)

        if self.startup_action == "restore_last":
            last_project_path = self.app_settings.value("Paths/last_project_dir", "")
            if last_project_path and Path(last_project_path).exists():
                self._perform_project_load(Path(last_project_path))
            else:
                self.stacked_layout.setCurrentIndex(0)
        elif self.startup_action == "new_project":
            self._handle_start_new_project_action()
        else:
            self.stacked_layout.setCurrentIndex(0)

    def _setup_main_app_tabs_container(self):
        self.main_tab_widget = QTabWidget(self)
        self.main_tab_widget.setMinimumSize(800, 500)
        self.main_tab_widget.setTabPosition(QTabWidget.TabPosition.North)
        self.main_tab_widget.setMovable(True)

        self.config_tab = ConfigWidget(self.default_config_loader, self)
        self.config_tab.configurations_updated.connect(self._on_app_configurations_updated)
        self.config_tab.save_configuration_to_file_requested.connect(
            self._on_save_standalone_config_requested
        )
        self.config_tab.help_requested.connect(self.request_help)
        # Connect engine selection signal from Config Widget (single source of truth)
        self.config_tab.engine_selection_changed.connect(self._on_config_engine_changed)
        self.main_tab_widget.addTab(self.config_tab, qta.icon("fa5s.cogs"), "")

        self.data_management_tab = DataManagementWidget(
            self,
            preferences_manager=self.preferences_manager,
            config_manager=self.default_config_loader,
        )
        self.data_management_tab.project_data_updated.connect(self._on_project_data_model_updated)
        self.data_management_tab.status_message_updated.connect(self.show_status_message)
        self.main_tab_widget.addTab(self.data_management_tab, qta.icon("fa5s.database"), "")

        # Debug: Check what we're passing to OptimizationWidget
        logger.debug(f"Passing config_manager: {type(self.default_config_loader)}")
        logger.debug(f"Passing parent: {type(self)}")
        self.optimization_tab = OptimizationWidget(self.default_config_loader, self)
        self.optimization_tab.optimization_completed.connect(self._on_optimization_run_completed)
        self.optimization_tab.open_configuration_requested.connect(self._open_configuration_tab)
        self.optimization_tab.representative_mmp_calculated.connect(
            self._on_representative_mmp_updated
        )
        self.optimization_tab.engine_type_change_requested.connect(
            self._on_engine_type_change_requested
        )
        self.config_tab.configurations_updated.connect(
            self.optimization_tab.on_configurations_updated
        )
        self.main_tab_widget.addTab(self.optimization_tab, qta.icon("fa5s.rocket"), "")

        self.analysis_tab = AnalysisWidget(self.default_config_loader, self)
        self.analysis_tab.help_requested.connect(self.request_help)
        self.main_tab_widget.addTab(self.analysis_tab, qta.icon("fa5s.chart-line"), "")

        self.ai_assistant_tab = AIAssistantWidget(
            self.app_settings, self.preferences_manager, self.default_config_loader, self
        )
        self.ai_assistant_tab.request_context_data.connect(self._provide_context_to_ai_assistant)
        self.ai_assistant_tab.request_all_parameters.connect(self._provide_all_parameters_to_ai)
        self.main_tab_widget.addTab(self.ai_assistant_tab, qta.icon("fa5s.robot"), "")

    def _transition_to_main_app_view(self, focus_tab_index: int = 0):
        self.stacked_layout.setCurrentIndex(1)

        # Log geometry information for debugging
        screen = QApplication.primaryScreen()
        if screen:
            screen_geometry = screen.availableGeometry()
        else:
            # Fallback for screen geometry if primaryScreen() returns None
            screen_geometry = self.geometry()
            logger.warning("QApplication.primaryScreen() returned None, using window geometry as fallback.")
        
        current_geometry = self.geometry()
        min_size = self.minimumSize()

        logger.debug(
            f"Transition geometry - Screen: {screen_geometry.width()}x{screen_geometry.height()}, "
            f"Current: {current_geometry.width()}x{current_geometry.height()}, "
            f"Min: {min_size.width()}x{min_size.height()}"
        )

        # Calculate splitter sizes that respect screen constraints
        available_width = min(self.width(), screen_geometry.width())
        help_panel_width = min(350, available_width - 400)  # Ensure minimum 400px for main content
        main_content_width = available_width - help_panel_width

        self.main_splitter.setSizes([main_content_width, help_panel_width])

        self.main_tab_widget.setCurrentIndex(focus_tab_index)
        self.save_project_action.setEnabled(True)
        self.save_project_as_action.setEnabled(True)
        self.close_project_action.setEnabled(True)
        self.new_project_action.setEnabled(True)
        logger.info(f"Transitioned to main app view, tab index: {focus_tab_index}")

    def _create_actions(self):
        icon_color = "#000000"

        self.new_project_action = QAction(qta.icon("fa5s.file", color=icon_color), "", self)
        self.new_project_action.setShortcut(QKeySequence.StandardKey.New)
        self.new_project_action.triggered.connect(self._handle_start_new_project_action)
        self.open_project_action = QAction(qta.icon("fa5s.folder-open", color=icon_color), "", self)
        self.open_project_action.setShortcut(QKeySequence.StandardKey.Open)
        self.open_project_action.triggered.connect(self._handle_load_project_action)
        self.save_project_action = QAction(qta.icon("fa5s.save", color=icon_color), "", self)
        self.save_project_action.setShortcut(QKeySequence.StandardKey.Save)
        self.save_project_action.triggered.connect(self._handle_save_project_action)
        self.save_project_as_action = QAction(
            qta.icon("fa5s.file-export", color=icon_color), "", self
        )
        self.save_project_as_action.setShortcut(QKeySequence.StandardKey.SaveAs)
        self.save_project_as_action.triggered.connect(self._handle_save_project_as_action)
        self.close_project_action = QAction(
            qta.icon("fa5s.window-close", color=icon_color), "", self
        )
        self.close_project_action.triggered.connect(self._handle_close_project_action)
        self.exit_action = QAction(qta.icon("fa5s.sign-out-alt", color=icon_color), "", self)
        self.exit_action.setShortcut(QKeySequence.StandardKey.Quit)
        self.exit_action.triggered.connect(self.close)
        self.settings_action = QAction(qta.icon("fa5s.cog", color=icon_color), "", self)
        self.settings_action.triggered.connect(self._show_preferences_dialog)
        self.generate_report_action = QAction(qta.icon("fa5s.file-pdf", color=icon_color), "", self)
        self.generate_report_action.triggered.connect(self._show_report_config_dialog)
        self.about_action = QAction(qta.icon("fa5s.info-circle", color=icon_color), "", self)
        self.about_action.triggered.connect(self._show_about_dialog)
        self.about_qt_action = QAction("", self)
        self.about_qt_action.triggered.connect(QApplication.aboutQt)

        self.associate_phd_action = QAction(qta.icon("fa5s.link", color=icon_color), "", self)
        self.associate_phd_action.triggered.connect(self._handle_associate_phd_files)
        self.remove_association_action = QAction(
            qta.icon("fa5s.unlink", color=icon_color), "", self
        )
        self.remove_association_action.triggered.connect(self._handle_remove_association)

        self.scientific_justification_action = QAction(
            qta.icon("fa5s.flask", color=icon_color), "", self
        )
        self.scientific_justification_action.triggered.connect(
            self._show_scientific_justification_dialog
        )

    def _show_scientific_justification_dialog(self):
        dialog = QDialog(self)
        dialog.setWindowTitle(self.tr("Scientific Justification"))
        layout = QVBoxLayout(dialog)
        content = ScientificJustificationWidget(dialog)
        layout.addWidget(content)
        dialog.show()

    def _create_menu_bar(self):
        self.menu_bar = self.menuBar()
        self.file_menu = self.menu_bar.addMenu("")
        self.file_menu.addAction(self.new_project_action)
        self.file_menu.addAction(self.open_project_action)
        self.file_menu.addSeparator()
        self.file_menu.addAction(self.save_project_action)
        self.file_menu.addAction(self.save_project_as_action)
        self.file_menu.addSeparator()
        self.file_menu.addAction(self.close_project_action)
        self.file_menu.addSeparator()
        self.file_menu.addAction(self.exit_action)

        self.edit_menu = self.menu_bar.addMenu("")
        self.edit_menu.addAction(self.settings_action)

        self.tools_menu = self.menu_bar.addMenu("")
        self.tools_menu.addAction(self.generate_report_action)
        self.tools_menu.addAction(self.associate_phd_action)
        self.tools_menu.addAction(self.remove_association_action)

        self.help_menu = self.menu_bar.addMenu("")

        icon_color = "#000000"

        self.show_help_action = QAction(
            qta.icon("fa5s.question-circle", color=icon_color), "", self
        )
        self.show_help_action.setCheckable(True)
        self.show_help_action.triggered.connect(self._toggle_help_panel)
        self.help_menu.addAction(self.show_help_action)
        self.help_menu.addSeparator()
        self.scientific_justification_action = QAction(
            qta.icon("fa5s.flask", color=icon_color), "", self
        )
        self.scientific_justification_action.triggered.connect(
            self._show_scientific_justification_dialog
        )
        self.help_menu.addAction(self.scientific_justification_action)
        self.help_menu.addSeparator()
        self.help_menu.addAction(self.about_action)
        self.help_menu.addAction(self.about_qt_action)

    def _toggle_help_panel(self):
        self.help_panel.setVisible(not self.help_panel.isVisible())

    def _create_status_bar(self):
        self.status_bar = QStatusBar(self)
        self.setStatusBar(self.status_bar)
        if PSUTIL_AVAILABLE:
            self.cpu_label = QLabel("")
            self.status_bar.addPermanentWidget(self.cpu_label)
            self.ram_label = QLabel("")
            self.status_bar.addPermanentWidget(self.ram_label)
        if GPUTIL_AVAILABLE and GPUtil.getGPUs():
            self.gpu_label = QLabel("")
            self.status_bar.addPermanentWidget(self.gpu_label)
        self.status_progress_bar = QProgressBar(self)
        self.status_progress_bar.setRange(0, 100)
        self.status_progress_bar.setVisible(False)
        self.status_progress_bar.setFixedSize(150, 16)
        self.status_bar.addPermanentWidget(self.status_progress_bar)

    @pyqtSlot(int, str)
    def _update_report_progress(self, value: int, message: str):
        self.show_status_message(message, 0)
        if value > 0 and value < 100:
            self.status_progress_bar.setVisible(True)
            self.status_progress_bar.setValue(value)
        else:
            self.status_progress_bar.setVisible(False)

    @pyqtSlot(str, str, str)
    def _show_message_box(self, title, text, msg_type):
        if msg_type == "info":
            QMessageBox.information(self, title, text)
        elif msg_type == "warning":
            QMessageBox.warning(self, title, text)
        elif msg_type == "critical":
            QMessageBox.critical(self, title, text)

    def retranslateUi(self):
        self.setWindowTitle(QApplication.applicationName())
        self._update_window_title()

        # Actions
        self.new_project_action.setText(self.tr("&New Project"))
        self.open_project_action.setText(self.tr("&Open Project..."))
        self.save_project_action.setText(self.tr("&Save Project"))
        self.save_project_as_action.setText(self.tr("Save Project &As..."))
        self.close_project_action.setText(self.tr("&Close Project"))
        self.exit_action.setText(self.tr("E&xit"))
        self.settings_action.setText(self.tr("&Preferences..."))
        self.generate_report_action.setText(self.tr("&Generate Report..."))
        self.about_action.setText(self.tr("&About ") + QApplication.applicationName())
        self.about_qt_action.setText(self.tr("About &Qt"))
        self.associate_phd_action.setText(self.tr("Associate .phd Files"))
        self.remove_association_action.setText(self.tr("Remove .phd Association"))
        self.show_help_action.setText(self.tr("Show Help Panel"))
        self.scientific_justification_action.setText(self.tr("Scientific Justification"))

        # Menus
        self.file_menu.setTitle(self.tr("&File"))
        self.edit_menu.setTitle(self.tr("&Edit"))
        self.tools_menu.setTitle(self.tr("&Tools"))
        self.help_menu.setTitle(self.tr("&Help"))

        # Tabs
        self.main_tab_widget.setTabText(0, self.tr("1. Configuration"))
        self.main_tab_widget.setTabText(1, self.tr("2. Data Management"))
        self.main_tab_widget.setTabText(2, self.tr("3. Optimization"))
        self.main_tab_widget.setTabText(3, self.tr("4. Analysis"))
        self.main_tab_widget.setTabText(4, self.tr("5. AI Assistant"))
        self.main_tab_widget.setTabText(5, self.tr("6. Application Logs"))

        # Status Bar
        if not self.status_bar.currentMessage():
            self.show_status_message(self.tr("Ready."), 5000)
        if PSUTIL_AVAILABLE:
            self.cpu_label.setText(self.tr(" CPU: --% "))
            self.ram_label.setText(self.tr(" RAM: --% "))
        if GPUTIL_AVAILABLE and hasattr(self, "gpu_label"):
            self.gpu_label.setText(self.tr(" GPU: --% "))

    def changeEvent(self, event: QEvent):
        if event.type() == QEvent.Type.LanguageChange:
            self.retranslateUi()
        super().changeEvent(event)

    def resizeEvent(self, event):
        """Handle resize events to ensure geometry stays within screen constraints."""
        super().resizeEvent(event)

        # Validate geometry after resize
        screen_geometry = QApplication.primaryScreen().availableGeometry()
        current_geometry = self.geometry()

        # Check if window is outside screen bounds
        if (
            current_geometry.right() > screen_geometry.right()
            or current_geometry.bottom() > screen_geometry.bottom()
            or current_geometry.left() < screen_geometry.left()
            or current_geometry.top() < screen_geometry.top()
        ):
            # Correct geometry if needed
            new_x = max(
                screen_geometry.left(),
                min(current_geometry.x(), screen_geometry.right() - current_geometry.width()),
            )
            new_y = max(
                screen_geometry.top(),
                min(current_geometry.y(), screen_geometry.bottom() - current_geometry.height()),
            )

            if new_x != current_geometry.x() or new_y != current_geometry.y():
                self.move(new_x, new_y)
                logger.debug(f"Corrected window position during resize: {new_x}, {new_y}")

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
        if PSUTIL_AVAILABLE:
            self.cpu_label.setText(f"{self.tr('CPU')}: {psutil.cpu_percent():.0f}% ")
            self.ram_label.setText(f"{self.tr('RAM')}: {psutil.virtual_memory().percent:.0f}% ")
        if GPUTIL_AVAILABLE and hasattr(self, "gpu_label"):
            try:
                gpu = GPUtil.getGPUs()[0]
                self.gpu_label.setText(f"{self.tr('GPU')}: {gpu.load * 100:.0f}% ")
            except Exception:
                self.gpu_label.setText(f"{self.tr('GPU')}: {self.tr('N/A')} ")

    def _create_tool_bar(self):
        pass

    def _update_window_title(self):
        title = QApplication.applicationName()
        if self.current_project_name:
            title += f" - {self.current_project_name}"
        elif self.project_file_path:
            title += f" - {self.project_file_path.name}"
        else:
            title += self.tr(" - New Project")
        title += "*" if self.is_project_modified else ""
        self.setWindowTitle(title)

    def set_project_modified(self, modified: bool = True):
        if self.is_project_modified != modified:
            self.is_project_modified = modified
            self._update_window_title()

    def show_status_message(self, message: str, timeout: int = 3000):
        self.status_bar.showMessage(message, timeout)
        logger.debug(f"Status: {message}")

    def _handle_start_new_project_action(self):
        if self._confirm_unsaved_changes():
            logger.info("Starting new project.")
            self.project_file_path = None
            self._initialize_project_data_and_configs()
            all_configs = {
                EconomicParameters.__name__: deepcopy(self.current_economic_params),
                EORParameters.__name__: deepcopy(self.current_eor_params),
                OperationalParameters.__name__: deepcopy(self.current_operational_params),
                ProfileParameters.__name__: deepcopy(self.current_profile_params),
                GeneticAlgorithmParams.__name__: deepcopy(self.current_ga_params),
                BayesianOptimizationParams.__name__: deepcopy(self.current_bo_params),
            }
            self.config_tab.update_configurations(all_configs)
            self.data_management_tab.clear_all_project_data()
            self.optimization_tab.update_engine(None)
            self.optimization_tab.update_project_data([], None)
            self.analysis_tab.update_analyzer_and_engine(None, None)
            self._update_window_title()
            self.set_project_modified(False)
            self._transition_to_main_app_view(focus_tab_index=1)
            self.show_status_message(
                self.tr("New project started. Configure parameters and load data."), 5000
            )

    def _handle_load_project_action(self):
        if not self._confirm_unsaved_changes():
            return
        last_dir = self.app_settings.value("Paths/last_project_dir", str(Path.home()))
        filepath_str, _ = QFileDialog.getOpenFileName(
            self,
            self.tr("Open Project File"),
            last_dir,
            self.tr("CO2 EOR Project Files (*.tphd);;All Files (*)"),
        )
        if not filepath_str:
            return
        filepath = Path(filepath_str)
        self.app_settings.setValue("Paths/last_project_dir", str(filepath.parent))
        self.show_status_message(
            self.tr("Loading project: {filepath_name}...").format(filepath_name=filepath.name), 0
        )
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        QTimer.singleShot(50, lambda: self._perform_project_load(filepath))

    def _perform_project_load(self, filepath: Path):
        try:
            project_data_dict = load_project_from_tphd(filepath)
            if project_data_dict is None:
                raise IOError("Failed to load or parse project file.")
            self._initialize_project_data_and_configs(from_project_load=True)
            self.current_well_data = project_data_dict.get("well_data_list", [])
            self.current_reservoir_data = project_data_dict.get("reservoir_data")
            self.current_pvt_properties = self._ensure_dataclass_instance(
                project_data_dict.get("pvt_properties"), PVTProperties
            )
            self.current_mmp_value = project_data_dict.get("mmp_value")
            self.current_project_name = project_data_dict.get("project_name")
            self.current_economic_params = self._ensure_dataclass_instance(
                project_data_dict.get("economic_parameters"), EconomicParameters
            )
            self.current_eor_params = self._ensure_dataclass_instance(
                project_data_dict.get("eor_parameters"), EORParameters
            )
            self.current_operational_params = self._ensure_dataclass_instance(
                project_data_dict.get("operational_parameters"), OperationalParameters
            )
            self.current_profile_params = self._ensure_dataclass_instance(
                project_data_dict.get("profile_parameters"), ProfileParameters
            )
            self.current_ga_params = self._ensure_dataclass_instance(
                project_data_dict.get("ga_parameters"), GeneticAlgorithmParams
            )
            self.current_bo_params = self._ensure_dataclass_instance(
                project_data_dict.get("bo_parameters"), BayesianOptimizationParams
            )

            self._loaded_project_data = {
                "configs": {
                    EconomicParameters.__name__: self.current_economic_params,
                    EORParameters.__name__: self.current_eor_params,
                    OperationalParameters.__name__: self.current_operational_params,
                    ProfileParameters.__name__: self.current_profile_params,
                    GeneticAlgorithmParams.__name__: self.current_ga_params,
                    BayesianOptimizationParams.__name__: self.current_bo_params,
                },
                "uq_parameters": project_data_dict.get("uq_parameters", []),
                "sensitivity_results": project_data_dict.get("sensitivity_results"),
                "optimization_results": project_data_dict.get("optimization_results"),
                "ui_state": project_data_dict.get("ui_state", {}),
                "recent_projects": project_data_dict.get("recent_projects", []),
            }

            self.project_file_path = filepath
            self.set_project_modified(False)
            self._update_window_title()

            self.progress_dialog = QProgressDialog(
                self.tr("Updating UI..."), self.tr("Cancel"), 0, 5, self
            )
            self.progress_dialog.setWindowTitle(self.tr("Loading Project"))
            self.progress_dialog.setWindowModality(Qt.WindowModality.WindowModal)
            self.progress_dialog.setValue(0)
            self.progress_dialog.show()

            QTimer.singleShot(0, self._update_ui_after_project_load)

            if self.stacked_layout.currentIndex() == 1:
                self._transition_to_main_app_view(
                    focus_tab_index=current_tab_index if "current_tab_index" in locals() else 0
                )
            else:
                self.stacked_layout.setCurrentIndex(
                    stacked_layout_index if "stacked_layout_index" in locals() else 0
                )

            self.overview_page.add_recent_project(str(filepath))
            self.show_status_message(
                self.tr("Project '{filepath_name}' loaded successfully.").format(
                    filepath_name=filepath.name
                ),
                5000,
            )
            logger.info(f"Project loaded: {filepath}")
        except Exception as e:
            logger.error(f"Failed to load project from {filepath}: {e}", exc_info=True)
            QMessageBox.critical(
                self,
                self.tr("Load Project Error"),
                self.tr("Could not load project file:\n{filepath}\n\nError: {e}").format(
                    filepath=filepath, e=e
                ),
            )
            self.project_file_path = None
            self._update_window_title()
        finally:
            QApplication.restoreOverrideCursor()

    def _update_ui_after_project_load(self):
        if not hasattr(self, "_loaded_project_data"):
            return

        data = self._loaded_project_data
        try:
            if self.progress_dialog:
                self.progress_dialog.setLabelText(self.tr("Updating configurations..."))
                self.progress_dialog.setValue(1)
            self.config_tab.update_configurations(data["configs"])
            QApplication.processEvents()

            if self.progress_dialog:
                self.progress_dialog.setLabelText(self.tr("Updating data management..."))
                self.progress_dialog.setValue(2)
            self.data_management_tab.load_project_data(
                {
                    "reservoir_data": self.current_reservoir_data,
                    "pvt_properties": self.current_pvt_properties,
                    "well_data_list": self.current_well_data,
                }
            )
            QApplication.processEvents()

            if self.progress_dialog:
                self.progress_dialog.setLabelText(self.tr("Updating engines and tabs..."))
                self.progress_dialog.setValue(3)
            self._update_engines_and_tabs(data)
            QApplication.processEvents()

            if self.progress_dialog:
                self.progress_dialog.setLabelText(self.tr("Restoring UI state..."))
                self.progress_dialog.setValue(4)
            self._restore_ui_state(data)
            QApplication.processEvents()

            if self.progress_dialog:
                self.progress_dialog.setLabelText(self.tr("Updating recent projects..."))
                self.progress_dialog.setValue(5)
            self._update_recent_projects(data)
            QApplication.processEvents()

            self._update_graphs_after_load()

            if data.get("optimization_results"):
                self._on_optimization_run_completed(data.get("optimization_results"))

        except Exception as e:
            logger.error(f"Error updating UI after project load: {e}", exc_info=True)
        finally:
            if self.progress_dialog:
                self.progress_dialog.close()
                self.progress_dialog = None
            self._cleanup_loaded_data()

    def _update_engines_and_tabs(self, data):
        try:
            self._reinitialize_engines_and_analysis_tabs(
                skip_calculations=True,
                sensitivity_results=data.get("sensitivity_results"),
                uq_parameters=data.get("uq_parameters"),
            )

            optimization_results = data.get("optimization_results")
            if optimization_results and self.optimisation_engine_instance:
                self.optimisation_engine_instance.results = optimization_results
                self.optimization_tab.current_results = optimization_results
                logger.info("Manually set optimization results on engine and tab.")

            uq_results = data.get("uq_results")
            if uq_results and self.analysis_tab.uq_engine:
                self.analysis_tab.uq_engine.results = uq_results
                logger.info("Manually set UQ results on UQEngine.")

        except Exception as e:
            logger.error(f"Error updating engines and tabs: {e}", exc_info=True)

    def _restore_ui_state(self, data):
        """Restore UI state from loaded data."""
        try:
            ui_state = data["ui_state"]
            if ui_state:
                current_tab_index = ui_state.get("current_tab_index", 0)
                splitter_sizes = ui_state.get("splitter_sizes", [self.width(), 0])
                help_panel_visible = ui_state.get("help_panel_visible", False)
                stacked_layout_index = ui_state.get("stacked_layout_index", 1)

                self.main_tab_widget.setCurrentIndex(current_tab_index)
                self.main_splitter.setSizes(splitter_sizes)
                if help_panel_visible:
                    self.help_panel.show()
                else:
                    self.help_panel.hide()
                self.stacked_layout.setCurrentIndex(stacked_layout_index)
        except Exception as e:
            logger.error(f"Error restoring UI state: {e}", exc_info=True)

    def _update_recent_projects(self, data):
        """Update recent projects list."""
        try:
            recent_projects = data["recent_projects"]
            if recent_projects:
                self.overview_page.set_recent_projects(recent_projects)
        except Exception as e:
            logger.error(f"Error updating recent projects: {e}", exc_info=True)

    def _cleanup_loaded_data(self):
        """Clean up temporary loaded project data."""
        if hasattr(self, "_loaded_project_data"):
            del self._loaded_project_data

    def _update_graphs_after_load(self):
        """Explicitly tell optimization and analysis tabs to update their graphs."""
        logger.info("Requesting graph updates in Optimization and Analysis tabs.")
        try:
            self.optimization_tab.update_graphs()
            self.analysis_tab.update_graphs()
        except Exception as e:
            logger.error(f"Error updating graphs after load: {e}", exc_info=True)

    def _confirm_unsaved_changes(self) -> bool:
        if not self.is_project_modified:
            return True
        reply = QMessageBox.question(
            self,
            self.tr("Unsaved Changes"),
            self.tr("The current project has unsaved changes. Do you want to save them?"),
            QMessageBox.StandardButton.Save
            | QMessageBox.StandardButton.Discard
            | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        if reply == QMessageBox.StandardButton.Save:
            return self._handle_save_project_action()
        return reply != QMessageBox.StandardButton.Cancel

    def _handle_save_project_action(self) -> bool:
        if self.project_file_path:
            return self._perform_project_save(self.project_file_path)
        else:
            return self._handle_save_project_as_action()

    def _handle_save_project_as_action(self) -> bool:
        last_dir = (
            str(self.project_file_path.parent)
            if self.project_file_path
            else self.app_settings.value("Paths/last_project_dir", str(Path.home()))
        )
        filepath_str, _ = QFileDialog.getSaveFileName(
            self,
            self.tr("Save Project As"),
            last_dir,
            self.tr("CO2 EOR Project Files (*.tphd);;All Files (*)"),
        )
        if not filepath_str:
            return False
        new_filepath = Path(filepath_str).with_suffix(".tphd")
        if self._perform_project_save(new_filepath):
            self.project_file_path = new_filepath
            self.app_settings.setValue("Paths/last_project_dir", str(new_filepath.parent))
            self._update_window_title()
            return True
        return False

    def _prepare_results_for_saving(
        self, results: Optional[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        if not results:
            return None

        clean_results = {}

        # List of keys to copy directly
        serializable_keys = [
            "optimized_params_final_clipped",
            "objective_function_value",
            "optimized_profiles",
            "final_metrics",
            "method",
            "ga_statistics",
            "bo_statistics",
            "pso_cost_history",
            "evaluated_points",
            "diverse_points_for_bo",
            "ga_full_results_for_hybrid",
        ]

        for key in serializable_keys:
            if key in results:
                if key == "ga_full_results_for_hybrid":
                    clean_results[key] = self._prepare_results_for_saving(results[key])
                else:
                    clean_results[key] = results[key]

        if "bayes_opt_obj" in results:
            bo_obj = results["bayes_opt_obj"]
            if hasattr(bo_obj, "res"):
                clean_results["bayes_opt_obj_res"] = bo_obj.res

        return clean_results

    def _perform_project_save(self, filepath: Path) -> bool:
        self.show_status_message(
            self.tr("Saving project to {filepath_name}...").format(filepath_name=filepath.name), 0
        )
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            optimization_results_to_save = self._prepare_results_for_saving(
                self.optimisation_engine_instance.results
                if self.optimisation_engine_instance
                else None
            )
            data_to_save = {
                "schema_version": "1.1",
                "application_version": QApplication.applicationVersion(),
                "project_name": self.current_project_name or filepath.stem,
                "well_data_list": self.current_well_data,
                "reservoir_data": self.current_reservoir_data,
                "pvt_properties": self.current_pvt_properties,
                "mmp_value": self.current_mmp_value,
                "economic_parameters": self.current_economic_params,
                "eor_parameters": self.current_eor_params,
                "operational_parameters": self.current_operational_params,
                "profile_parameters": self.current_profile_params,
                "ga_parameters": self.current_ga_params,
                "bo_parameters": self.current_bo_params,
                "optimization_results": optimization_results_to_save,
                "sensitivity_results": self.sensitivity_analyzer_instance.sensitivity_run_data
                if self.sensitivity_analyzer_instance
                else None,
                "uq_parameters": self.analysis_tab.get_uq_parameters(),
                "ui_state": {
                    "current_tab_index": self.main_tab_widget.currentIndex(),
                    "splitter_sizes": self.main_splitter.sizes(),
                    "help_panel_visible": self.help_panel.isVisible(),
                    "stacked_layout_index": self.stacked_layout.currentIndex(),
                },
                "recent_projects": self.overview_page.get_recent_projects(),
            }

            if save_project_to_tphd(data_to_save, filepath):
                if not self.current_project_name:
                    self.current_project_name = data_to_save["project_name"]
                self.set_project_modified(False)
                self.overview_page.add_recent_project(str(filepath))
                self.show_status_message(
                    self.tr("Project saved: {filepath_name}").format(filepath_name=filepath.name),
                    5000,
                )
                logger.info(f"Project saved to {filepath}")
                return True
            raise IOError("Project saving function returned failure.")
        except Exception as e:
            logger.error(f"Failed to save project to {filepath}: {e}", exc_info=True)
            QMessageBox.critical(
                self,
                self.tr("Save Project Error"),
                self.tr("Could not save project file:\n{filepath}\n\nError: {e}").format(
                    filepath=filepath, e=e
                ),
            )
            return False
        finally:
            QApplication.restoreOverrideCursor()

    def _handle_close_project_action(self):
        if self._confirm_unsaved_changes():
            logger.info("Closing current project.")
            self.project_file_path = None
            self._initialize_project_data_and_configs()
            all_configs = {
                EconomicParameters.__name__: self.current_economic_params,
                EORParameters.__name__: self.current_eor_params,
                OperationalParameters.__name__: self.current_operational_params,
                ProfileParameters.__name__: self.current_profile_params,
                GeneticAlgorithmParams.__name__: self.current_ga_params,
                BayesianOptimizationParams.__name__: self.current_bo_params,
            }
            self.config_tab.update_configurations(all_configs)
            self.data_management_tab.clear_all_project_data()
            self.optimization_tab.update_engine(None)
            self.optimization_tab.update_project_data([], None)
            self.analysis_tab.update_analyzer_and_engine(None, None)
            self.stacked_layout.setCurrentIndex(0)
            self.save_project_action.setEnabled(False)
            self.save_project_as_action.setEnabled(False)
            self.close_project_action.setEnabled(False)
            self._update_window_title()
            self.show_status_message(self.tr("Project closed."), 5000)

    def _handle_quick_start_action(self):
        import json
        from PyQt6.QtWidgets import (
            QDialog,
            QVBoxLayout,
            QListWidget,
            QTextBrowser,
            QDialogButtonBox,
            QLabel,
            QSplitter,
        )

        dialog = QDialog(self)
        dialog.setWindowTitle(self.tr("Quick Start Demo Scenarios"))
        dialog.setMinimumSize(600, 400)

        layout = QVBoxLayout(dialog)

        try:
            with open("config/demo_data.json", "r") as f:
                demo_data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            QMessageBox.critical(
                self, self.tr("Error"), self.tr("Could not load demo data: ") + str(e)
            )
            return

        list_widget = QListWidget()
        for item in demo_data:
            list_widget.addItem(item["name"])

        text_browser = QTextBrowser()
        text_browser.setOpenExternalLinks(True)

        def update_details():
            selected_item_name = list_widget.currentItem().text()
            for item in demo_data:
                if item["name"] == selected_item_name:
                    details = f"<h3>{item['name']}</h3>"
                    details += f"<b>{self.tr('Overview')}:</b><p>{item['overview']}</p>"
                    details += f"<b>{self.tr('Reference')}:</b><p>{item['reference']}</p>"
                    text_browser.setHtml(details)
                    break

        list_widget.currentItemChanged.connect(update_details)

        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        button_box.button(QDialogButtonBox.StandardButton.Ok).setText(self.tr("Load"))
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)

        layout.addWidget(QLabel(self.tr("Select a demo scenario to load:")))
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(list_widget)
        splitter.addWidget(text_browser)
        splitter.setSizes([200, 400])

        layout.addWidget(splitter)
        layout.addWidget(button_box)

        if demo_data:
            list_widget.setCurrentRow(0)
            update_details()

        if dialog.exec() == QDialog.DialogCode.Accepted:
            selected_item_name = list_widget.currentItem().text()
            for item in demo_data:
                if item["name"] == selected_item_name:
                    self._load_demo_data(item["parameters"])
                    break

    def _load_demo_data(self, params: dict):
        logger.info(f"Loading quick start demo data into UI: {params}")
        if self._confirm_unsaved_changes():
            self.project_file_path = None
            self._initialize_project_data_and_configs()

            try:
                if "eor" in params:
                    self.current_eor_params = self._ensure_dataclass_instance(
                        params["eor"], EORParameters
                    )

                self.data_management_tab.populate_fields_from_demo(params)

            except Exception as e:
                logger.error(f"Error processing demo data: {e}", exc_info=True)
                QMessageBox.critical(
                    self, self.tr("Error"), self.tr("Failed to process demo data.")
                )
                return

            all_configs = {
                EORParameters.__name__: deepcopy(self.current_eor_params),
                EconomicParameters.__name__: deepcopy(self.current_economic_params),
                OperationalParameters.__name__: deepcopy(self.current_operational_params),
                ProfileParameters.__name__: deepcopy(self.current_profile_params),
                GeneticAlgorithmParams.__name__: deepcopy(self.current_ga_params),
                BayesianOptimizationParams.__name__: deepcopy(self.current_bo_params),
            }
            self.config_tab.update_configurations(all_configs)

            self._update_window_title()
            self.set_project_modified(True)
            self._transition_to_main_app_view(focus_tab_index=0)  # configuration tab
            self.show_status_message(
                self.tr("Demo data loaded. Please review and click 'Generate Project Data'."), 8000
            )

    def _handle_open_recent_project(self, filepath_str: str):
        filepath = Path(filepath_str)
        if not filepath.exists():
            QMessageBox.warning(
                self,
                self.tr("File Not Found"),
                self.tr("The project file was not found:\n{filepath}").format(filepath=filepath),
            )
            self.overview_page.add_recent_project("")
            return

        if self._confirm_unsaved_changes():
            self.show_status_message(
                self.tr("Loading recent project: {filepath_name}...").format(
                    filepath_name=filepath.name
                ),
                0,
            )
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            QTimer.singleShot(50, lambda: self._perform_project_load(filepath))

    def _show_preferences_dialog(self):
        """Show the preferences dialog and apply changes if accepted."""
        from ui.dialogs.preferences_dialog import PreferencesDialog

        i18n_manager = None
        pref_manager = None
        if hasattr(self, "preferences_manager") and self.preferences_manager:
            i18n_manager = self.preferences_manager.i18n_manager
            pref_manager = self.preferences_manager

        dialog = PreferencesDialog(self, i18n_manager, pref_manager)
        dialog.preferences_applied.connect(self._on_preferences_applied)

        if dialog.exec() == QDialog.DialogCode.Accepted:
            logger.info("Preferences dialog accepted, changes applied")
        else:
            logger.info("Preferences dialog cancelled")

    def _on_preferences_applied(self):
        logger.info("Preferences were applied, updating UI if needed")

    def _on_general_preferences_changed(self, preferences):
        logger.info("General preferences changed")

        if (
            hasattr(self, "current_auto_save_interval")
            and self.current_auto_save_interval != preferences.auto_save_interval
        ):
            self._setup_auto_save_timer()

        self.current_auto_save_interval = preferences.auto_save_interval

    def _on_display_preferences_changed(self, preferences):
        logger.info("Display preferences changed")
        self._update_display_units()
        self._apply_font_preference(preferences.font_size)
        self._apply_graphing_preference(
            preferences.graph_anti_aliasing, preferences.graph_smooth_lines
        )
        self._apply_tooltip_preference(preferences.show_tooltips)
        self._apply_animation_preference(preferences.animation_enabled)

    def _apply_font_preference(self, font_size: int):
        font = QApplication.font()
        if font_size > 0:
            font.setPointSize(font_size)
        else:
            default_font = QFont()
            font.setPointSize(default_font.pointSize())
        QApplication.setFont(font)

    def _apply_graphing_preference(self, anti_aliasing: bool, smooth_lines: bool):
        self.graphing_config = {
            "displaylogo": False,
            "modeBarButtonsToRemove": ["toImage", "sendDataToCloud"],
            "doubleClick": "reset+autosize",
            "showTips": True,
            "scrollZoom": True,
            "antialias": anti_aliasing,
            "smooth": smooth_lines,
        }
        logger.info(
            f"Graphing preferences updated: AA={anti_aliasing}, Smooth Lines={smooth_lines}"
        )

    def _apply_tooltip_preference(self, show_tooltips: bool):
        """Enable or disable tooltips throughout the application."""
        logger.info(f"Tooltip preference updated: Show={show_tooltips}")

    def _apply_animation_preference(self, enabled: bool):
        logger.info(f"Animation preference updated: Enabled={enabled}")

    def _on_units_preferences_changed(self, preferences):
        logger.info("Units preferences changed")
        self._update_display_units()

    def _on_language_preferences_changed(self, preferences):
        logger.info("Language preferences changed signal received by MainWindow.")

    def _on_advanced_preferences_changed(self, preferences):
        logger.info("Advanced preferences changed")

    def _on_preferences_reset(self):
        logger.info("Preferences reset to defaults")
        self._update_display_units()

    def _update_display_units(self):
        logger.info("Updating display units based on preferences")
        if self.preferences_manager:
            unit_system = self.preferences_manager.display.unit_system.value
            logger.info(f"Unit system changed to: {unit_system}")

        self._update_all_unit_displays()

    def _setup_auto_save_timer(self):
        if self.auto_save_timer:
            self.auto_save_timer.stop()
            self.auto_save_timer.deleteLater()
            self.auto_save_timer = None

        interval = self.preferences_manager.general.auto_save_interval
        if interval > 0:
            self.auto_save_timer = QTimer(self)
            self.auto_save_timer.setInterval(
                interval * 60 * 1000
            )  # Convert minutes to milliseconds
            self.auto_save_timer.timeout.connect(self._auto_save_project)
            self.auto_save_timer.start()
            logger.info(f"Auto-save timer started with {interval} minute interval")
        else:
            logger.info("Auto-save disabled")

    def _auto_save_project(self):
        if self.is_project_modified and self.project_file_path:
            try:
                logger.info(f"Auto-saving project to {self.project_file_path}...")
                if self._perform_project_save(self.project_file_path):
                    logger.info("Project auto-saved successfully")
                else:
                    logger.error("Auto-save failed.")
            except Exception as e:
                logger.error(f"Auto-save failed: {e}")

    def _update_all_unit_displays(self):
        logger.info("Updating all unit displays")

    def _show_report_config_dialog(self):
        from ui.dialogs.report_config_dialog import ReportConfigDialog

        if not self.current_project_name:
            from PyQt6.QtWidgets import QInputDialog

            project_name, ok = QInputDialog.getText(
                self,
                self.tr("Project Name"),
                self.tr("Please enter a name for your project:"),
                text=self.tr("Unnamed Project"),
            )
            if not ok or not project_name.strip():
                return
            self.current_project_name = project_name.strip()
            self.set_project_modified(True)
            self._update_window_title()

        current_config = {}
        project_name = self.current_project_name

        data_availability = {
            "optimization_results": bool(
                self.optimisation_engine_instance and self.optimisation_engine_instance.results
            ),
            "sensitivity_analysis": bool(
                self.sensitivity_analyzer_instance
                and self.sensitivity_analyzer_instance.sensitivity_run_data
            ),
            "uq_analysis": bool(
                hasattr(self.analysis_tab, "uq_engine")
                and self.analysis_tab.uq_engine
                and hasattr(self.analysis_tab.uq_engine, "results")
                and self.analysis_tab.uq_engine.results
            ),
        }

        config = ReportConfigDialog.configure_report(
            current_config, project_name, data_availability, self
        )

        if config:

            def generate_report_task():
                try:
                    self.progress_updated.emit(0, self.tr("Preparing report data..."))
                    report_data = self._prepare_report_data()

                    ai_features = config.get("ai_features", {})
                    if any(ai_features.values()):
                        self.progress_updated.emit(0, self.tr("Generating AI summaries..."))
                        self._generate_ai_summaries(ai_features, report_data)

                    self.progress_updated.emit(0, self.tr("Generating report..."))
                    success = self.report_generator.generate_report(
                        report_data, config["output_path"], config
                    )
                    if success:
                        self.progress_updated.emit(
                            100,
                            self.tr("Report generated: {output_path}").format(
                                output_path=config["output_path"]
                            ),
                        )
                        self.show_message.emit(
                            self.tr("Success"),
                            self.tr("Report generated successfully:\n{output_path}").format(
                                output_path=config["output_path"]
                            ),
                            "info",
                        )
                    else:
                        self.progress_updated.emit(0, self.tr("Report generation failed"))
                        self.show_message.emit(
                            self.tr("Error"),
                            self.tr("Report generation failed. Check logs for details."),
                            "warning",
                        )
                except Exception as e:
                    self.progress_updated.emit(
                        0, self.tr("Report error: {error}").format(error=str(e))
                    )
                    self.show_message.emit(
                        self.tr("Error"),
                        self.tr("Report generation failed: {error}").format(error=str(e)),
                        "critical",
                    )

            import threading

            thread = threading.Thread(target=generate_report_task)
            thread.daemon = True
            thread.start()

    def _plotly_fig_to_base64(self, fig: go.Figure) -> str:
        if not fig:
            return ""
        try:
            img_bytes = to_image(fig, format="png", width=800, height=500, scale=2)
            return f"data:image/png;base64,{base64.b64encode(img_bytes).decode('utf-8')}"
        except Exception as e:
            logger.error(f"Failed to convert plot to image: {e}")
            return ""

    def _prepare_report_data(self) -> Dict[str, Any]:
        app = QApplication.instance()
        session_id = app.property("session_id") if app and app.property("session_id") else "N/A"

        report_data = {
            "project_name": self.current_project_name or self.tr("Unnamed Project"),
            "reservoir_data": asdict(self.current_reservoir_data)
            if self.current_reservoir_data
            else {},
            "pvt_data": asdict(self.current_pvt_properties) if self.current_pvt_properties else {},
            "economic_parameters": asdict(self.current_economic_params)
            if self.current_economic_params
            else {},
            "eor_parameters": asdict(self.current_eor_params) if self.current_eor_params else {},
            "operational_parameters": asdict(self.current_operational_params)
            if self.current_operational_params
            else {},
            "profile_parameters": asdict(self.current_profile_params)
            if self.current_profile_params
            else {},
            "well_data": [asdict(well) for well in self.current_well_data]
            if self.current_well_data
            else [],
            "session_id": session_id,
        }

        report_charts = {}
        if self.optimisation_engine_instance:
            report_data["recovery_model"] = self.optimisation_engine_instance.recovery_model
            if self.optimisation_engine_instance.results:
                report_data["optimization_results"] = self.optimisation_engine_instance.results
                report_data["validation_report"] = (
                    self.optimisation_engine_instance.get_validation_report()
                )
                report_data["optimization_stats"] = self.optimisation_engine_instance.results.get(
                    "ga_statistics"
                ) or self.optimisation_engine_instance.results.get("bo_statistics")

                # Use pre-generated charts from the results dictionary
                if "charts" in self.optimisation_engine_instance.results:
                    for chart_name, chart_fig in self.optimisation_engine_instance.results[
                        "charts"
                    ].items():
                        report_charts[chart_name] = self._plotly_fig_to_base64(chart_fig)

        if self.sensitivity_analyzer_instance and self.sensitivity_analyzer_instance.report_runs:
            last_run = self.sensitivity_analyzer_instance.report_runs[-1]
            report_data["sensitivity_results"] = last_run["df"]
            df = last_run["df"]
            objective_col = last_run["context"].get("objective", "npv")
            charts["sensitivity_tornado"] = self._plotly_fig_to_base64(
                self.sensitivity_analyzer_instance.plot_tornado_chart(df, objective_col)
            )

        if (
            hasattr(self.analysis_tab, "uq_engine")
            and self.analysis_tab.uq_engine
            and hasattr(self.analysis_tab.uq_engine, "results")
            and self.analysis_tab.uq_engine.results
        ):
            report_data["uq_results"] = self.analysis_tab.uq_engine.results
            mc_results_df = self.analysis_tab.uq_engine.results.get("mc_results_df")
            if mc_results_df is not None:
                objective_col = self.analysis_tab.uq_engine.objective_column or "npv"
                charts["uq_distribution"] = self._plotly_fig_to_base64(
                    self.analysis_tab.uq_engine.plot_mc_results(mc_results_df, objective_col)
                )

        if hasattr(self.analysis_tab, "dca_results") and self.analysis_tab.dca_results:
            report_data["dca_results"] = self.analysis_tab.dca_analyzer.generate_dca_report_data(
                self.analysis_tab.dca_results
            ).generate_dca_report_data()

        report_data["charts"] = charts

        return report_data

    def _show_about_dialog(self):
        QMessageBox.about(
            self,
            self.tr("About ") + QApplication.applicationName(),
            f"<h2>{QApplication.applicationName()}</h2><p>{self.tr('Version')}: {QApplication.applicationVersion()}</p><p>{self.tr('A CO₂ Enhanced Oil Recovery Suite.')}</p>",
        )

    @pyqtSlot(dict)
    def _on_project_data_model_updated(self, project_data_dict: Dict[str, Any]):
        logger.info("MainWindow: Received project_data_updated signal.")
        self.set_project_modified(True)
        self.current_well_data = project_data_dict.get("well_data_list", [])
        self.current_reservoir_data = project_data_dict.get("reservoir_data")
        self.current_pvt_properties = project_data_dict.get("pvt_properties")

        # Handle EOR parameters from data management widget (surrogate engine integration)
        eor_params_from_widget = project_data_dict.get("eor_parameters")
        if eor_params_from_widget and self.current_eor_params:
            # Update current EOR parameters with values from data management widget
            for key, value in eor_params_from_widget.__dict__.items():
                if hasattr(self.current_eor_params, key):
                    setattr(self.current_eor_params, key, value)
            logger.info(
                f"MainWindow - Updated EOR parameters from widget: {list(eor_params_from_widget.__dict__.keys())}"
            )

        # Handle EOR parameters override (legacy support)
        eor_params_override = project_data_dict.get("eor_params_override")
        if eor_params_override and self.current_eor_params:
            # Update current EOR parameters with values from data management widget
            for key, value in eor_params_override.items():
                if hasattr(self.current_eor_params, key):
                    setattr(self.current_eor_params, key, value)
            logger.info(
                f"MainWindow - Updated EOR parameters with override: {list(eor_params_override.keys())}"
            )

        # Handle operational parameters from data management widget
        operational_params_from_widget = project_data_dict.get("operational_parameters")
        if operational_params_from_widget and self.current_operational_params:
            # Update current operational parameters with values from widget
            for key, value in operational_params_from_widget.__dict__.items():
                if hasattr(self.current_operational_params, key):
                    setattr(self.current_operational_params, key, value)
            logger.info(
                f"MainWindow - Updated operational parameters from widget"
            )

        # Handle economic parameters from data management widget
        economic_params_from_widget = project_data_dict.get("economic_parameters")
        if economic_params_from_widget and self.current_economic_params:
            # Update current economic parameters with values from widget
            for key, value in economic_params_from_widget.__dict__.items():
                if hasattr(self.current_economic_params, key):
                    setattr(self.current_economic_params, key, value)
            logger.info(
                f"MainWindow - Updated economic parameters from widget"
            )

        is_data_finalization = project_data_dict.get("is_data_finalization", False)
        self._reinitialize_engines_and_analysis_tabs(skip_calculations=is_data_finalization)
        self.show_status_message(self.tr("Core project data models updated."), 3000)

    @pyqtSlot(dict)
    def _on_app_configurations_updated(self, new_configs: Dict[str, Any]):
        logger.info("MainWindow state updating from ConfigWidget.")
        self.set_project_modified(True)
        self.current_economic_params = new_configs.get(
            EconomicParameters.__name__, self.current_economic_params
        )
        self.current_eor_params = new_configs.get(EORParameters.__name__, self.current_eor_params)
        self.current_operational_params = new_configs.get(
            OperationalParameters.__name__, self.current_operational_params
        )
        self.current_profile_params = new_configs.get(
            ProfileParameters.__name__, self.current_profile_params
        )
        self.current_ga_params = new_configs.get(
            GeneticAlgorithmParams.__name__, self.current_ga_params
        )
        self.current_bo_params = new_configs.get(
            BayesianOptimizationParams.__name__, self.current_bo_params
        )
        # DEBUG: Log the received EOR parameters
        eor_params = new_configs.get(EORParameters.__name__)
        if eor_params:
            logger.info(
                f"MainWindow - Received EOR Parameters - Injection Scheme: '{eor_params.injection_scheme}', "
                f"WAG Ratio: {eor_params.WAG_ratio}"
            )

        self._reinitialize_engines_and_analysis_tabs()

    @pyqtSlot(float)
    def _on_representative_mmp_updated(self, mmp_value: float):
        logger.info(f"MainWindow received representative MMP value: {mmp_value:.2f} psi")
        if self.current_mmp_value != mmp_value:
            self.current_mmp_value = mmp_value
            self.set_project_modified(True)
            self._reinitialize_engines_and_analysis_tabs()
            self.show_status_message(
                self.tr("Optimization Engine updated with MMP = {mmp_value:.1f} psi").format(
                    mmp_value=mmp_value
                ),
                5000,
            )

    @pyqtSlot(dict)
    def _on_save_standalone_config_requested(self, config_data: Dict[str, Any]):
        logger.info(f"ConfigWidget saved standalone configuration.")

    @pyqtSlot(str)
    def _on_engine_type_change_requested(self, engine_type: str):
        """Handle engine type change request from optimization widget."""
        try:
            logger.info(f"Engine type change requested: {engine_type}")

            # Validate engine type
            valid_types = ["simple", "detailed", "surrogate"]
            if engine_type not in valid_types:
                raise ValueError(f"Invalid engine type: {engine_type}. Must be one of {valid_types}")

            # Update the current advanced engine params
            if not hasattr(self, 'current_advanced_engine_params'):
                from core.data_models import AdvancedEngineParams
                self.current_advanced_engine_params = AdvancedEngineParams()

            # Set both old (for backward compat) and new fields
            self.current_advanced_engine_params.use_simple_physics = (engine_type == "simple")
            self.current_advanced_engine_params.engine_type = engine_type

            # Reinitialize engines with new settings
            self._reinitialize_engines_and_analysis_tabs()

            # Show status message
            self.show_status_message(
                f"Engine switched to {engine_type}. Optimization engine recreated.",
                5000
            )

            logger.info(f"Engine successfully switched to {engine_type}")

        except Exception as e:
            error_msg = f"Failed to switch engine: {str(e)}"
            logger.error(error_msg, exc_info=True)
            QMessageBox.critical(self, "Engine Switch Error", error_msg)

    @pyqtSlot(str)
    def _on_config_engine_changed(self, engine_type: str):
        """Handle engine type change from Config Widget (single source of truth).

        This is called when the user changes the engine selection in the Config Widget.
        The Config Widget is now the authoritative location for engine selection.
        """
        try:
            logger.info(f"Engine type changed from Config Widget: {engine_type}")

            # Validate engine type
            valid_types = ["simple", "detailed", "surrogate"]
            if engine_type not in valid_types:
                raise ValueError(f"Invalid engine type: {engine_type}. Must be one of {valid_types}")

            # Update the current advanced engine params
            if not hasattr(self, 'current_advanced_engine_params'):
                from core.data_models import AdvancedEngineParams
                self.current_advanced_engine_params = AdvancedEngineParams()

            # Set both old (for backward compat) and new fields
            self.current_advanced_engine_params.use_simple_physics = (engine_type == "simple")
            self.current_advanced_engine_params.engine_type = engine_type

            # Update data management widget (disable/enable fields based on engine)
            if hasattr(self, 'data_management_tab'):
                self.data_management_tab.set_engine_type(engine_type)

            # Update optimization widget - notify of engine change
            # Note: OptimizationWidget no longer has its own engine selection UI
            # It reads from MainWindow.current_advanced_engine_params
            if hasattr(self, 'optimization_tab'):
                self.optimization_tab.on_engine_type_changed(engine_type)

            # Reinitialize engines if data is loaded
            if self.current_reservoir_data:
                self._reinitialize_engines_and_analysis_tabs()

            # Show status message
            self.show_status_message(
                f"Engine switched to {engine_type}. All widgets updated.",
                5000
            )

            logger.info(f"Engine successfully switched to {engine_type} from Config Widget")

        except Exception as e:
            error_msg = f"Failed to switch engine from Config Widget: {str(e)}"
            logger.error(error_msg, exc_info=True)
            QMessageBox.critical(self, "Engine Switch Error", error_msg)

    def _reinitialize_engines_and_analysis_tabs(
        self,
        skip_calculations: bool = False,
        sensitivity_results: Optional[List[Dict[str, Any]]] = None,
        uq_parameters: Optional[List[Dict[str, Any]]] = None,
    ):
        logger.info(
            f"Re-initializing engines and analysis tabs (skip_calculations={skip_calculations})."
        )
        self.optimisation_engine_instance = None
        self.sensitivity_analyzer_instance = None
        if self.current_reservoir_data and self.current_pvt_properties:
            try:
                # DEBUG: Log reservoir data before creating engine
                if self.current_reservoir_data and hasattr(
                    self.current_reservoir_data, "eos_model"
                ):
                    logger.debug(
                        f"About to create engine with EOS model type: {type(self.current_reservoir_data.eos_model)}"
                    )
                    logger.debug(f"EOS model value: {self.current_reservoir_data.eos_model}")
                    if hasattr(self.current_reservoir_data.eos_model, "__dict__"):
                        logger.debug(
                            f"EOS model dict: {self.current_reservoir_data.eos_model.__dict__}"
                        )

                self.optimisation_engine_instance = OptimizationEngine(
                    reservoir=deepcopy(self.current_reservoir_data),
                    pvt=deepcopy(self.current_pvt_properties),
                    eor_params_instance=deepcopy(self.current_eor_params),
                    ga_params_instance=deepcopy(self.current_ga_params),
                    bo_params_instance=deepcopy(self.current_bo_params),
                    economic_params_instance=deepcopy(self.current_economic_params),
                    operational_params_instance=deepcopy(self.current_operational_params),
                    profile_params_instance=deepcopy(self.current_profile_params),
                    advanced_engine_params_instance=deepcopy(self.current_advanced_engine_params),
                    well_data_list=self.current_well_data,
                    mmp_init_override=self.current_mmp_value,
                )

                # DEBUG: Log the EOR parameters passed to the engine
                if self.current_eor_params:
                    logger.info(
                        f"MainWindow - Engine Initialized with EOR Parameters - Injection Scheme: '{self.current_eor_params.injection_scheme}', "
                        f"WAG Ratio: {self.current_eor_params.WAG_ratio}"
                    )

                logger.info("OptimizationEngine instance created.")
                QApplication.processEvents()

                if not skip_calculations and self.current_mmp_value is None:
                    logger.info(
                        "No MMP override present. Triggering self-calculation for the Optimization Engine."
                    )
                    self.optimisation_engine_instance.calculate_mmp()

                self.sensitivity_analyzer_instance = SensitivityAnalyzer(
                    engine=self.optimisation_engine_instance
                )
                QApplication.processEvents()

                if sensitivity_results:
                    self.sensitivity_analyzer_instance.sensitivity_run_data = sensitivity_results
                logger.info("SensitivityAnalyzer instance re-created successfully.")
            except Exception as e:
                logger.error(f"Failed to create engine/analyzer: {e}", exc_info=True)
                QMessageBox.critical(
                    self,
                    self.tr("Engine Error"),
                    self.tr("Could not initialize core engines: {e}").format(e=e),
                )
        else:
            logger.warning("Cannot create engines: Missing reservoir or PVT data.")

        self.optimization_tab.update_engine(self.optimisation_engine_instance)
        QApplication.processEvents()

        self.optimization_tab.update_project_data(
            self.current_well_data, self.current_pvt_properties
        )
        self.analysis_tab.update_analyzer_and_engine(
            self.sensitivity_analyzer_instance,
            self.optimisation_engine_instance,
            self.current_well_data,
        )
        QApplication.processEvents()

        if uq_parameters:
            self.analysis_tab.set_uq_parameters(uq_parameters)
            QApplication.processEvents()

    @pyqtSlot()
    def _open_configuration_tab(self):
        self.main_tab_widget.setCurrentIndex(0)
        self.show_status_message(self.tr("Configuration panel opened."), 3000)

    @pyqtSlot(dict)
    def _on_optimization_run_completed(self, results: Dict[str, Any]):
        logger.info("MainWindow: Optimization run completed.")
        self.set_project_modified(True)
        self.show_status_message(self.tr("Optimization complete."), 5000)
        if self.optimisation_engine_instance and self.optimisation_engine_instance.results:
            try:
                self.sensitivity_analyzer_instance = SensitivityAnalyzer(
                    engine=self.optimisation_engine_instance
                )

                well_data_for_analysis = list(self.current_well_data)

                optimized_profiles = self.optimisation_engine_instance.results.get(
                    "optimized_profiles"
                )
                if optimized_profiles:
                    resolution = (
                        self.optimisation_engine_instance.operational_params.time_resolution
                    )
                    oil_stb_key = f"{resolution}_oil_stb"
                    oil_stb = optimized_profiles.get(oil_stb_key)

                    if oil_stb is not None and len(oil_stb) > 0:
                        rate_multiplier = 1.0
                        if resolution == "quarterly":
                            rate_multiplier = 4.0
                        elif resolution == "monthly":
                            rate_multiplier = 12.0
                        elif resolution == "weekly":
                            rate_multiplier = 52.0

                        rate_stb_per_year = oil_stb * rate_multiplier

                        time_years = np.arange(1, len(oil_stb) + 1)
                        if resolution == "quarterly":
                            time_years = time_years / 4.0
                        elif resolution == "monthly":
                            time_years = time_years / 12.0
                        elif resolution == "weekly":
                            time_years = time_years / 52.0

                        field_well_data = WellData(
                            name="Field (Optimized)",
                            depths=np.array([]),
                            properties={"time": time_years, "rate": rate_stb_per_year},
                            units={"time": "years", "rate": "STB/year"},
                            metadata={"source": "Optimization"},
                        )
                        well_data_for_analysis.append(field_well_data)
                        logger.info(
                            "Created pseudo-well 'Field (Optimized)' for DCA from optimization results."
                        )

                self.analysis_tab.update_analyzer_and_engine(
                    self.sensitivity_analyzer_instance,
                    self.optimisation_engine_instance,
                    well_data_for_analysis,
                )
                logger.info("Analysis tab updated with new optimization results as base.")
            except Exception as e:
                logger.error(f"Error updating Analysis Tab post-optimization: {e}", exc_info=True)

    @pyqtSlot(str)
    def _provide_context_to_ai_assistant(self, context_req: str):
        context_data = {}
        if self.optimisation_engine_instance:
            context_data["optimization_engine_state"] = self.optimisation_engine_instance.results
        if self.current_reservoir_data:
            context_data["reservoir_summary"] = {
                "grid_dims": self.current_reservoir_data.runspec.get("DIMENSIONS")
                if self.current_reservoir_data.runspec
                else "N/A",
                "ooip": self.current_reservoir_data.ooip_stb,
            }
        self.ai_assistant_tab.set_context_data(context_req, context_data)

    @pyqtSlot()
    def _provide_all_parameters_to_ai(self):
        import dataclasses

        parameters = {
            "Economic Parameters": [f.name for f in dataclasses.fields(EconomicParameters)],
            "EOR Parameters": [f.name for f in dataclasses.fields(EORParameters)],
            "Operational Parameters": [f.name for f in dataclasses.fields(OperationalParameters)],
            "Profile Parameters": [f.name for f in dataclasses.fields(ProfileParameters)],
            "Genetic Algorithm": [f.name for f in dataclasses.fields(GeneticAlgorithmParams)],
            "Bayesian Optimization": [
                f.name for f in dataclasses.fields(BayesianOptimizationParams)
            ],
        }
        self.ai_assistant_tab.show_parameter_selection(parameters)

    def save_window_settings(self):
        self.app_settings.setValue("MainWindow/geometry", self.saveGeometry())
        self.app_settings.setValue("MainWindow/state", self.saveState())
        self.app_settings.setValue("MainWindow/splitterSizes", self.main_splitter.saveState())
        logger.debug("Window settings saved.")

    def load_window_settings(self):
        if geom := self.app_settings.value("MainWindow/geometry"):
            self.restoreGeometry(geom)
            screen_geometry = QApplication.primaryScreen().availableGeometry()
            current_geometry = self.geometry()

            min_size = self.minimumSize()
            min_width = min_size.width()
            min_height = min_size.height()

            # Enhanced geometry validation with detailed logging
            logger.debug(
                f"Window settings load - Screen: {screen_geometry.width()}x{screen_geometry.height()}, "
                f"Current: {current_geometry.width()}x{current_geometry.height()}, "
                f"Min: {min_width}x{min_height}"
            )

            # Ensure window fits within screen bounds
            new_width = max(min_width, min(current_geometry.width(), screen_geometry.width()))
            new_height = max(min_height, min(current_geometry.height(), screen_geometry.height()))

            # Ensure window position is valid
            new_x = max(
                screen_geometry.left(),
                min(current_geometry.x(), screen_geometry.right() - new_width),
            )
            new_y = max(
                screen_geometry.top(),
                min(current_geometry.y(), screen_geometry.bottom() - new_height),
            )

            # Apply corrected geometry if needed
            if (
                new_x != current_geometry.x()
                or new_y != current_geometry.y()
                or new_width != current_geometry.width()
                or new_height != current_geometry.height()
            ):
                logger.debug(
                    f"Correcting window geometry from {current_geometry.width()}x{current_geometry.height()} to {new_width}x{new_height}"
                )
                self.setGeometry(new_x, new_y, new_width, new_height)

        if state := self.app_settings.value("MainWindow/state"):
            self.restoreState(state)

        if splitter_state := self.app_settings.value("MainWindow/splitterSizes"):
            self.main_splitter.restoreState(splitter_state)
        else:
            # Set reasonable default splitter sizes
            screen_width = QApplication.primaryScreen().availableGeometry().width()
            main_content_width = max(
                800, screen_width - 350
            )  # Ensure minimum 800px for main content
            help_panel_width = min(350, screen_width - 800)
            self.main_splitter.setSizes([main_content_width, help_panel_width])

        if self.stacked_layout.currentIndex() == 0:
            self.help_panel.hide()

        logger.debug("Window settings loaded with enhanced geometry validation.")

    def closeEvent(self, event: QCloseEvent):
        if self._confirm_unsaved_changes():
            self.save_window_settings()
            logger.info("Closing application.")
            event.accept()
        else:
            event.ignore()

    def _handle_associate_phd_files(self) -> None:
        try:
            executable_path = Path(sys.executable)
            icon_path = get_ui_assets_dir() / "main_ico.ico"

            manager = FileAssociationManager()
            success = manager.associate_phd_files(executable_path, icon_path)

            if success:
                QMessageBox.information(
                    self,
                    self.tr("File Association"),
                    self.tr("Successfully associated .phd files with this application."),
                )
            else:
                QMessageBox.warning(
                    self,
                    self.tr("File Association"),
                    self.tr(
                        "Failed to associate .phd files. This feature is only available on Windows."
                    ),
                )

        except Exception as e:
            logger.error(f"Error associating .phd files: {e}")
            QMessageBox.critical(
                self,
                self.tr("File Association Error"),
                self.tr("An error occurred while trying to associate .phd files:\n\n{e}").format(
                    e=e
                ),
            )

    def _handle_remove_association(self) -> None:
        try:
            manager = FileAssociationManager()
            success = manager.remove_association()

            if success:
                QMessageBox.information(
                    self,
                    self.tr("File Association"),
                    self.tr("Successfully removed .phd file association."),
                )
            else:
                QMessageBox.warning(
                    self,
                    self.tr("File Association"),
                    self.tr(
                        "Failed to remove .phd file association. This feature is only available on Windows."
                    ),
                )

        except Exception as e:
            logger.error(f"Error removing .phd file association: {e}")
            QMessageBox.critical(
                self,
                self.tr("File Association Error"),
                self.tr(
                    "An error occurred while trying to remove .phd file association:\n\n{e}"
                ).format(e=e),
            )
