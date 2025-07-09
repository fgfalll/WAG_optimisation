"""
main.py: Main application entry point for the CO2 EOR Suite.

This script initializes the PyQt6 application, sets up critical services like
logging and configuration management, handles internationalization, and launches
the main window. The initialization logic is contained within the main()
function to ensure a predictable startup sequence.
"""

import sys
import logging
import os
from pathlib import Path
from typing import Optional
import time

try:
    package_root = Path(__file__).resolve().parent
    workspace_root = package_root.parent
    if str(workspace_root) not in sys.path:
        sys.path.insert(0, str(workspace_root))
except NameError:
    package_root = Path.cwd()
    workspace_root = package_root.parent
    if str(workspace_root) not in sys.path:
        sys.path.insert(0, str(workspace_root))

from co2eor_optimizer.config_manager import ConfigManager, ConfigNotLoadedError
from PyQt6.QtWidgets import QApplication, QMessageBox
from PyQt6.QtCore import QTranslator, QLocale, QSettings, QCoreApplication, Qt

APP_NAME = "CO2EORSuite"
APP_VERSION = "0.1.0-alpha"
ORGANIZATION_NAME = "PetroTheoHydeoPhD"
ORGANIZATION_DOMAIN = "petrotheohydeo.com"

logger = logging.getLogger(__name__)

def setup_logging(config: Optional[ConfigManager] = None) -> None:
    """Configures the global logging system based on loaded configuration."""
    log_config = {}
    if config:
        log_config = config.get("Logging", {})

    log_level_str = log_config.get("level", "INFO")
    log_format_str = log_config.get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    log_file_str = log_config.get("log_file", "co2_eor_suite.log")
    if not log_file_str: log_file_str = "co2_eor_suite.log"

    log_level = getattr(logging, log_level_str.upper(), logging.INFO)
    log_file_path = package_root / log_file_str

    try:
        log_file_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        log_file_path = Path(log_file_path.name)
        logging.warning(f"Could not create log directory. Logging to {log_file_path.resolve()}. Error: {e}")

    file_handler = logging.FileHandler(log_file_path, mode='a', encoding='utf-8')
    console_handler = logging.StreamHandler(sys.stdout)
    
    file_handler.setLevel(log_level)
    console_handler.setLevel(log_level)
    
    formatter = logging.Formatter(log_format_str)
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    main_logger = logging.getLogger(__name__)
    main_logger.propagate = False
    main_logger.setLevel(log_level)
    main_logger.addHandler(file_handler)
    main_logger.addHandler(console_handler)
    
    global logger
    logger = main_logger


class CO2EORApplication(QApplication):
    def __init__(self, argv: list[str]):
        super().__init__(argv)
        self.translator: Optional[QTranslator] = None

        QCoreApplication.setApplicationName(APP_NAME)
        QCoreApplication.setApplicationVersion(APP_VERSION)
        QCoreApplication.setOrganizationName(ORGANIZATION_NAME)
        QCoreApplication.setOrganizationDomain(ORGANIZATION_DOMAIN)

    def load_translator(self, locale_str: Optional[str] = None) -> None:
        if self.translator:
            self.removeTranslator(self.translator)

        self.translator = QTranslator(self)
        locale_to_use = locale_str or QLocale.system().name()  # "en_US", "uk_UA"
        locale_short = locale_to_use.split('_')[0]
        
        # Translations are inside the package directory
        translation_file = package_root / "translations" / f"app_{locale_short}.qm"
        logger.info(f"Attempting to load translation for locale '{locale_short}' from {translation_file}")

        if self.translator.load(str(translation_file)):
            self.installTranslator(self.translator)
            logger.info(f"Successfully loaded translation for '{locale_short}'.")
        else:
            if locale_short != "en":
                logger.warning(f"Translation file not found or failed to load. Defaulting to English.")
        
        self.setProperty("current_locale_short", locale_short)


def timed_import_main_window():
    """
    Imports MainWindow directly in the main thread.
    Returns the MainWindow class on success, or raises an exception on failure.
    """
    try:
        start_time = time.time()
        from co2eor_optimizer.ui.main_window import MainWindow
        elapsed = time.time() - start_time
        logger.info(f"MainWindow imported in {elapsed:.2f} seconds")
        return MainWindow
    except Exception as e:
        raise e


def main() -> None:
    """
    Main function to initialize and run the application.
    Orchestrates the startup sequence:
    1. Create the QApplication instance.
    2. Load configurations.
    3. Set up logging.
    4. Set up QSettings for UI persistence.
    5. Handle internationalization.
    6. Create and show the MainWindow.
    7. Start the event loop.
    """
    QApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts)
    
    app = CO2EORApplication(sys.argv)

    # Create a temporary config manager just for setting up initial logging.
    # The MainWindow will create its own instance for handling application defaults.
    temp_config_loader = ConfigManager(config_dir_path=str(package_root / "config"), require_config=False, autoload=True)

    setup_logging(temp_config_loader)
    logger.info(f"--- {APP_NAME} v{APP_VERSION} Started ---")
    logger.info(f"Package Root: {package_root}")
    logger.info(f"Workspace Root: {workspace_root} (added to sys.path)")
    if temp_config_loader.is_loaded:
        logger.info(f"Configurations loaded successfully from: {temp_config_loader.loaded_config_directory}")
    else:
        logger.warning("ConfigManager is using internal defaults. External JSON files were not loaded.")

    settings = QSettings(ORGANIZATION_NAME, APP_NAME)
    logger.info(f"Application settings are stored at: {settings.fileName()}")

    preferred_locale = settings.value("General/preferred_locale", None, type=str)
    app.load_translator(preferred_locale)

    logger.info("Attempting to import MainWindow with a 15-second timeout...")
    try:
        MainWindow = timed_import_main_window()
        logger.info("MainWindow imported successfully.")
    except Exception as e:
        logger.critical(f"Fatal Error: Failed to import or initialize MainWindow. The application cannot start. Error: {e}", exc_info=True)
        QMessageBox.critical(None, "Fatal Error", f"Failed to import a critical UI component:\n\n{e}\n\nPlease check the installation and log file for details.")
        sys.exit(1)

    logger.info("Initializing MainWindow instance...")
    main_window = MainWindow(app_settings=settings)
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
                int(screen_geometry.height() * 0.8)
            )
            logger.info("Setting default window geometry for first launch.")

    main_window.show()
    logger.info("MainWindow initialized and shown.")

    exit_code = app.exec()
    logger.info(f"--- {APP_NAME} Exited (Code: {exit_code}) ---")
    sys.exit(exit_code)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        if logger.handlers:
            logger.critical(f"An unhandled exception occurred in the main execution block: {e}", exc_info=True)
        else:
            print(f"FATAL ERROR: {e}", file=sys.stderr)
        if QApplication.instance():
            QMessageBox.critical(None, "Fatal Application Error", f"An unexpected error occurred and the application must close:\n\n{e}\n\nSee the log file for technical details.")
        sys.exit(1)