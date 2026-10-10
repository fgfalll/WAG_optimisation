"""
Application entry point for CO2 EOR Optimizer.
Initializes application configuration, logging, preferences, and the main UI window.
"""

from __future__ import annotations

import logging
import sys
import time
import uuid
import warnings
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ui.main_window import MainWindow

warnings.filterwarnings("ignore", category=UserWarning, message=".*pkg_resources is deprecated as an API.*")

from PyQt6.QtCore import QCoreApplication, QSettings, Qt
from PyQt6.QtWidgets import QApplication, QMessageBox

from ui.qt_log_handler import QtLogHandler
from utils.config_manager import ConfigManager
from utils.i18n_manager import I18nManager
from utils.multiprocess_logging import init_application_logging, shutdown_queue_logging
from utils.path_utils import get_app_root, get_config_dir, get_translations_dir
from utils.preferences_manager import initialize_preferences_manager

APP_NAME = "CO2 Optimiser PhD project"
APP_VERSION = "0.8.5-alpha"
ORGANIZATION_NAME = "TIndustials"
ORGANIZATION_DOMAIN = "https://tarascv.netlify.app/"

logger = logging.getLogger(__name__)


class CO2EORApplication(QApplication):
    """Custom QApplication with project-specific metadata."""

    def __init__(self, argv: list[str]):
        super().__init__(argv)
        QCoreApplication.setApplicationName(APP_NAME)
        QCoreApplication.setApplicationVersion(APP_VERSION)
        QCoreApplication.setOrganizationName(ORGANIZATION_NAME)
        QCoreApplication.setOrganizationDomain(ORGANIZATION_DOMAIN)


def timed_import_main_window() -> type[MainWindow]:
    """
    Import MainWindow and return the class, timing and logging the import duration.
    Preserves signature and behavior for startup smoke tests.
    """
    logger.debug("Entering timed_import_main_window")
    start_time = time.perf_counter()
    try:
        from ui.main_window import MainWindow

        elapsed = time.perf_counter() - start_time
        logger.info(f"MainWindow imported in {elapsed:.2f} seconds")
        return MainWindow
    except Exception as e:
        logger.critical(f"Exception during MainWindow import: {e}", exc_info=True)
        raise
    finally:
        logger.info("Exiting timed_import_main_window")


def _normalize_module_log_levels(intended_level: int) -> None:
    """Ensure submodules inherit root log level rather than overriding it during imports."""
    current_level = logging.getLogger().level
    if current_level != intended_level:
        logging.getLogger().setLevel(intended_level)
        logger.warning(
            f"Logging level was reset to {logging.getLevelName(current_level)} during imports. "
            f"Restored to {logging.getLevelName(intended_level)}."
        )

    for logger_name in list(logging.root.manager.loggerDict.keys()):
        if logger_name.startswith(("core", "ui", "analysis", "utils")):
            logging.getLogger(logger_name).setLevel(logging.NOTSET)


def main() -> None:
    """Application main entry point."""
    QCoreApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts)
    app = CO2EORApplication(sys.argv)

    session_id = str(uuid.uuid4())[:10]
    app.setProperty("session_id", session_id)

    app_root = get_app_root()
    config_dir = get_config_dir()
    config_loader = ConfigManager(
        config_dir_path=str(config_dir), require_config=False, autoload=True
    )

    qt_log_handler = QtLogHandler()
    init_application_logging(config_loader, session_id, qt_log_handler)

    logger.info(f"--- {APP_NAME} v{APP_VERSION} Started ---")
    logger.info(f"Session ID: {session_id}")
    logger.info(f"App Root: {app_root}")
    if config_loader.is_loaded:
        logger.info(f"Configurations loaded from: {config_loader.loaded_config_directory}")
    else:
        logger.warning("ConfigManager is using internal defaults. External JSON files were not loaded.")

    settings = QSettings(ORGANIZATION_NAME, APP_NAME)
    logger.info(f"Application settings stored at: {settings.fileName()}")

    i18n_manager = I18nManager(get_translations_dir())
    i18n_manager.set_application_instance(app)

    pref_manager = initialize_preferences_manager(APP_NAME, ORGANIZATION_NAME, i18n_manager)
    pref_manager.apply_language_preferences()

    intended_log_level = logging.getLogger().level
    try:
        MainWindow = timed_import_main_window()
        _normalize_module_log_levels(intended_log_level)

        main_window = MainWindow(  # type: ignore[call-arg]
            app_settings=settings,
            preferences_manager=pref_manager,
            qt_log_handler=qt_log_handler,
        )
        main_window.show()
    except Exception as e:
        logger.critical(
            f"Fatal Error: Failed to initialize MainWindow: {e}",
            exc_info=True,
        )
        QMessageBox.critical(
            None,
            "Fatal Error",
            f"Failed to initialize the main application window:\n\n{e}\n\nPlease check installation and log file.",
        )
        sys.exit(1)

    exit_code = app.exec()
    logger.info(f"--- {APP_NAME} Exited (Code: {exit_code}) ---")
    shutdown_queue_logging()
    sys.exit(exit_code)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        if logger.handlers:
            logger.critical(f"Unhandled exception in main execution block: {e}", exc_info=True)
        else:
            logging.critical(f"FATAL ERROR: {e}", exc_info=True)
        if QApplication.instance():
            QMessageBox.critical(
                None,
                "Fatal Application Error",
                f"An unexpected error occurred and the application must close:\n\n{e}\n\nSee log file for details.",
            )
        shutdown_queue_logging()
        sys.exit(1)
