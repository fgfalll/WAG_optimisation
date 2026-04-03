import sys
import logging
from pathlib import Path
from typing import Optional
import time

from path_utils import get_app_root, get_config_dir, get_logs_dir, get_translations_dir
from config_manager import ConfigManager, ConfigNotLoadedError
from utils.preferences_manager import initialize_preferences_manager
from utils.i18n_manager import I18nManager
from utils.multiprocess_logging import setup_queue_logging, shutdown_queue_logging
from PyQt6.QtWidgets import QApplication, QMessageBox
from PyQt6.QtCore import QSettings, QCoreApplication, Qt
import uuid

APP_NAME = "CO2 Optimiser PhD project"
APP_VERSION = "0.8.5-alpha"
ORGANIZATION_NAME = "TIndustials"
ORGANIZATION_DOMAIN = "https://tarascv.netlify.app/"

logger = logging.getLogger(__name__)


def setup_logging(
    config: Optional[ConfigManager] = None,
    session_id: Optional[str] = None,
    qt_handler: Optional[logging.Handler] = None,
) -> None:
    log_config = {}
    if config:
        log_config = config.get("Logging", {})

    log_format_str = log_config.get(
        "format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    log_file_str = log_config.get("log_file", "app_co2eor.log")
    if not log_file_str:
        log_file_str = "app_co2eor.log"

    session_logging = log_config.get("session_based", True)
    if session_logging:
        log_file_str = f"session_{session_id or str(uuid.uuid4())[:8]}.log"

    log_level_str = log_config.get("level", "WARNING").upper()
    log_level = getattr(logging, log_level_str, logging.DEBUG)

    logs_dir = get_logs_dir()
    try:
        logs_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logging.warning(
            f"Could not create logs directory. Logging to current directory. Error: {e}"
        )
        logs_dir = Path.cwd()

    log_file_path = logs_dir / log_file_str

    setup_queue_logging(log_file_path, log_level, log_format_str)

    if qt_handler:
        root_logger = logging.getLogger()
        root_logger.addHandler(qt_handler)

    logger.info(f"Logging reconfigured for multiprocess safety. File: {log_file_path}")


def teardown_logging() -> None:
    """Clean up logging on application exit."""
    shutdown_queue_logging()


class CO2EORApplication(QApplication):
    def __init__(self, argv: list[str]):
        super().__init__(argv)
        QCoreApplication.setApplicationName(APP_NAME)
        QCoreApplication.setApplicationVersion(APP_VERSION)
        QCoreApplication.setOrganizationName(ORGANIZATION_NAME)
        QCoreApplication.setOrganizationDomain(ORGANIZATION_DOMAIN)


def timed_import_main_window():
    logger.debug("Entering timed_import_main_window")
    try:
        start_time = time.time()
        logger.debug("Importing logging")
        import logging

        logger.debug("Importing sys")
        import sys

        logger.debug("Importing Path")
        from pathlib import Path

        logger.debug("Importing typing")
        from typing import Optional, Any, Dict, List

        logger.debug("Importing deepcopy")
        from copy import deepcopy

        logger.debug("Importing traceback")
        import traceback

        logger.debug("Importing asdict")
        from dataclasses import asdict

        logger.debug("Importing pandas")
        import pandas as pd

        logger.debug("Importing base64")
        import base64

        logger.debug("Importing io")
        import io

        logger.debug("Importing to_image")
        from plotly.io import to_image

        logger.debug("Importing plotly.graph_objects")
        import plotly.graph_objects as go

        logger.debug("Importing qtawesome")
        import qtawesome as qta

        logger.debug("Importing QtWidgets")
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
        )

        logger.debug("Importing QtGui")
        from PyQt6.QtGui import QAction, QIcon, QKeySequence, QCloseEvent, QFont

        logger.debug("Importing QtCore")
        from PyQt6.QtCore import Qt, pyqtSignal, QSettings, QTimer, pyqtSlot, QEvent

        logger.debug("Importing QPalette, QColor")
        from PyQt6.QtGui import QPalette, QColor

        logger.debug("Importing psutil")
        try:
            import psutil

            PSUTIL_AVAILABLE = True
        except ImportError:
            PSUTIL_AVAILABLE = False
            logging.warning("psutil library not found. CPU/RAM monitoring will be disabled.")
        logger.debug("Importing GPUtil")
        try:
            import GPUtil

            GPUTIL_AVAILABLE = True
        except ImportError:
            GPUTIL_AVAILABLE = False
            logging.warning("GPUtil library not found. GPU monitoring will be disabled.")

        logger.info("Importing project modules")
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
        )
        from core.optimisation_engine import OptimizationEngine
        from analysis.sensitivity_analyzer import SensitivityAnalyzer
        from analysis.uq_engine import UncertaintyQuantificationEngine
        from analysis.well_analysis import WellAnalysis
        from utils.file_association import FileAssociationManager
        from utils.preferences_manager import get_preferences_manager
        from ui.dialogs.preferences_dialog import PreferencesDialog
        from ui.workers.ai_query_worker import AIQueryWorker

        logger.info("Importing MainWindow")
        from ui.main_window import MainWindow

        elapsed = time.time() - start_time
        logger.info(f"MainWindow imported in {elapsed:.2f} seconds")
        return MainWindow
    except Exception as e:
        logger.critical(f"Exception during MainWindow import: {e}", exc_info=True)
        raise e
    finally:
        logger.info("Exiting timed_import_main_window")


def main() -> None:
    QApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts)
    app = CO2EORApplication(sys.argv)
    session_id = str(uuid.uuid4())[:10]
    app.setProperty("session_id", session_id)
    try:
        import qtawesome as qta

        logger.info("QtAwesome imported successfully.")
    except Exception as e:
        logger.warning(f"Failed to import QtAwesome: {e}. Icons may not display correctly.")

    app_root = get_app_root()
    config_dir = get_config_dir()

    temp_config_loader = ConfigManager(
        config_dir_path=str(config_dir), require_config=False, autoload=True
    )

    from ui.qt_log_handler import QtLogHandler

    qt_log_handler = QtLogHandler()

    setup_logging(temp_config_loader, session_id, qt_log_handler)
    logger.info(f"--- {APP_NAME} v{APP_VERSION} Started ---")
    logger.info(f"Session ID: {session_id}")
    logger.info(f"App Root: {app_root}")
    if temp_config_loader.is_loaded:
        logger.info(
            f"Configurations loaded successfully from: {temp_config_loader.loaded_config_directory}"
        )
    else:
        logger.warning(
            "ConfigManager is using internal defaults. External JSON files were not loaded."
        )

    settings = QSettings(ORGANIZATION_NAME, APP_NAME)
    logger.info(f"Application settings are stored at: {settings.fileName()}")

    translations_dir = get_translations_dir()
    i18n_manager = I18nManager(translations_dir)
    i18n_manager.set_application_instance(app)

    pref_manager = initialize_preferences_manager(APP_NAME, ORGANIZATION_NAME, i18n_manager)

    pref_manager.apply_language_preferences()

    startup_action = pref_manager.general.startup_action

    logger.info("Attempting to import MainWindow with a 15-second timeout...")
    try:
        # Capture the intended logging level before imports might mess it up
        intended_log_level = logging.getLogger().level
        
        MainWindow = timed_import_main_window()
        
        # Restore the intended logging level if it was changed during imports
        current_level = logging.getLogger().level
        if current_level != intended_log_level:
            logging.getLogger().setLevel(intended_log_level)
            logger.warning(f"Logging level was reset to {logging.getLevelName(current_level)} during imports. Restored to {logging.getLevelName(intended_log_level)}.")
        
        # Enforce that all application modules inherit the root logger's level
        # This fixes the issue where some modules (or 3rd party libs) might have explicitly set their logger level to INFO
        for logger_name in logging.root.manager.loggerDict:
            if logger_name.startswith(("core", "ui", "analysis", "utils")):
                logging.getLogger(logger_name).setLevel(logging.NOTSET)
            
        logger.info("MainWindow imported successfully.")
    except Exception as e:
        logger.critical(
            f"Fatal Error: Failed to import or initialize MainWindow. The application cannot start. Error: {e}",
            exc_info=True,
        )
        QMessageBox.critical(
            None,
            "Fatal Error",
            f"Failed to import a critical UI component:\n\n{e}\n\nPlease check the installation and log file for details.",
        )
        sys.exit(1)

    logger.info("Initializing MainWindow instance...")
    main_window = MainWindow(
        app_settings=settings,
        preferences_manager=pref_manager,
        startup_action=startup_action,
        qt_log_handler=qt_log_handler,
    )
    logger.info("MainWindow instance created.")

    if settings.contains("MainWindow/geometry"):
        main_window.restoreGeometry(settings.value("MainWindow/geometry"))
    else:
        screen = app.primaryScreen()
        if screen:
            screen_geometry = screen.availableGeometry()
            main_window.setGeometry(
                int(screen_geometry.width() * 0.1),
                int(screen_geometry.height() * 0.1),
                int(screen_geometry.width() * 0.8),
                int(screen_geometry.height() * 0.8),
            )
            logger.info("Setting default window geometry for first launch.")

    main_window.show()
    logger.info("MainWindow initialized and shown.")

    exit_code = app.exec()
    logger.info(f"--- {APP_NAME} Exited (Code: {exit_code}) ---")
    teardown_logging()
    sys.exit(exit_code)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        if logger.handlers:
            logger.critical(
                f"An unhandled exception occurred in the main execution block: {e}", exc_info=True
            )
        else:
            logging.critical(f"FATAL ERROR: {e}", exc_info=True)
        if QApplication.instance():
            QMessageBox.critical(
                None,
                "Fatal Application Error",
                f"An unexpected error occurred and the application must close:\n\n{e}\n\nSee the log file for technical details.",
            )
        teardown_logging()
        sys.exit(1)
