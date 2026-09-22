# utils/i18n_manager.py

import logging
from pathlib import Path
from typing import List, Tuple, Optional, Union
from PyQt6.QtCore import QTranslator, QLocale, QCoreApplication # QCoreApplication for app instance

logger = logging.getLogger(__name__)

class I18nManager:
    """
    Manages internationalization aspects of the application, such as
    listing available languages and facilitating language switching.
    """
    def __init__(self, translations_dir: Union[str, Path]):
        """
        Initializes the I18nManager.

        Args:
            translations_dir: Path to the directory containing .qm translation files.
        """
        self.translations_dir = Path(translations_dir)
        if not self.translations_dir.is_dir():
            logger.warning(
                f"Translations directory '{self.translations_dir}' not found. "
                "Language features may be limited."
            )
        self._app_instance: Optional[QCoreApplication] = None

    def set_application_instance(self, app: QCoreApplication):
        """
        Sets the QApplication instance, needed for installing translators.
        This should be called once the QApplication object is created.
        """
        self._app_instance = app

    def get_available_locales(self) -> List[Tuple[str, str]]:
        """
        Scans the translations directory for available .qm files and returns
        a list of (locale_code, language_name) tuples.
        Example: [('en', 'English'), ('uk', 'Українська')]

        Returns:
            A list of tuples (locale_code, human_readable_language_name).
        """
        available = []
        if not self.translations_dir.is_dir():
            return available

        for qm_file in self.translations_dir.glob("app_*.qm"):
            try:
                # Assuming "app_{locale_code}.qm" format, e.g., "app_en.qm"
                locale_code = qm_file.stem.split("_")[-1]
                if locale_code:
                    # Get the native language name from QLocale
                    # For "en", QLocale("en").nativeLanguageName() is "English"
                    # For "uk", QLocale("uk").nativeLanguageName() is "українська"
                    locale_obj = QLocale(locale_code)
                    language_name = locale_obj.nativeLanguageName().capitalize()
                    if language_name and language_name != locale_code: # Ensure a proper name was found
                        available.append((locale_code, language_name))
                    else:
                        # Fallback if nativeLanguageName is empty or just the code
                        available.append((locale_code, locale_code.upper())) 
            except Exception as e:
                logger.warning(f"Could not parse locale from file {qm_file.name}: {e}")
        
        # Ensure English is listed if available, perhaps prioritize it
        available.sort(key=lambda x: (x[0] != 'en', x[1])) # English first, then alphabetical
        if not available:
            logger.info("No .qm translation files found in translations directory.")
        return available

    def load_and_install_translator(self, locale_code: str) -> bool:
        """
        Loads and installs a QTranslator for the given locale code.
        Assumes the QApplication instance has been set via `set_application_instance`.

        Args:
            locale_code: The short locale code (e.g., "en", "uk").

        Returns:
            True if the translator was successfully loaded and installed, False otherwise.
        """
        if self._app_instance is None:
            logger.error("QApplication instance not set in I18nManager. Cannot install translator.")
            return False

        # Remove any existing translator first
        if hasattr(self._app_instance, '_active_translator') and self._app_instance._active_translator: # type: ignore
            self._app_instance.removeTranslator(self._app_instance._active_translator) # type: ignore
            logger.debug("Removed existing translator.")

        translator = QTranslator(self._app_instance)
        translation_file_name = f"app_{locale_code}.qm"
        translation_path = self.translations_dir / translation_file_name

        if translator.load(str(translation_path)):
            self._app_instance.installTranslator(translator)
            setattr(self._app_instance, '_active_translator', translator) # Store it for removal later
            logger.info(f"Successfully loaded and installed translation for locale '{locale_code}' from {translation_path}.")
            
            # Update QLocale for the application if necessary (affects number/date formatting)
            # QLocale.setDefault(QLocale(locale_code)) # This changes global default, be careful
            
            # Emit a signal or property change if UI elements need to react dynamically
            # For simplicity, often a restart prompt is easier for full UI translation updates.
            self._app_instance.setProperty("current_locale_short", locale_code)
            
            return True
        else:
            logger.warning(f"Could not load translation file for locale '{locale_code}': {translation_path}")
            # Try to load English as a fallback if the requested locale failed and wasn't English
            if locale_code != 'en':
                logger.info("Attempting to load fallback English translation 'app_en.qm'.")
                en_translator = QTranslator(self._app_instance)
                en_translation_path = self.translations_dir / "app_en.qm"
                if en_translator.load(str(en_translation_path)):
                    self._app_instance.installTranslator(en_translator)
                    setattr(self._app_instance, '_active_translator', en_translator)
                    self._app_instance.setProperty("current_locale_short", "en")
                    logger.info("Successfully loaded and installed fallback English translation.")
                    return True # Indicate success even if it's fallback
                else:
                    logger.warning("Fallback English translation 'app_en.qm' also not found.")
            return False

# --- Example Usage (Conceptual, as it interacts with QApplication) ---
if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    
    # Assume project root structure:
    # project_root/
    #   utils/i18n_manager.py
    #   translations/
    #     app_en.qm
    #     app_uk.qm
    
    # Create dummy .qm files for testing (in a real scenario, these are compiled from .ts)
    project_root_dir = Path(__file__).resolve().parent.parent
    test_translations_dir = project_root_dir / "translations"
    test_translations_dir.mkdir(exist_ok=True)
    (test_translations_dir / "app_en.qm").touch(exist_ok=True)
    (test_translations_dir / "app_uk.qm").touch(exist_ok=True)
    (test_translations_dir / "app_fr.qm").touch(exist_ok=True) # Another example

    i18n_mgr = I18nManager(translations_dir=test_translations_dir)
    
    available = i18n_mgr.get_available_locales()
    logger.info(f"Available locales found: {available}")
    assert ('en', 'English') in available
    assert ('uk', 'Українська') in available # QLocale should give native names
    assert ('fr', 'Français') in available # QLocale should give native names

    # To test load_and_install_translator, you'd need a QApplication instance:
    # from PyQt6.QtWidgets import QApplication
    # import sys
    # app = QApplication(sys.argv)
    # i18n_mgr.set_application_instance(app)
    # if i18n_mgr.load_and_install_translator('uk'):
    #     logger.info("Ukrainian translator loaded (simulated).")
    #     # In a real app, UI elements using self.tr() would now update if they are re-rendered
    #     # or if they are designed to react to locale change signals.
    # else:
    #     logger.error("Failed to load Ukrainian translator (simulated).")

    # Clean up dummy files
    # (test_translations_dir / "app_en.qm").unlink(missing_ok=True)
    # (test_translations_dir / "app_uk.qm").unlink(missing_ok=True)
    # (test_translations_dir / "app_fr.qm").unlink(missing_ok=True)
    # if test_translations_dir.exists() and not any(test_translations_dir.iterdir()):
    #      test_translations_dir.rmdir()
    
    logger.info("I18nManager example tests completed (conceptual for load/install).")