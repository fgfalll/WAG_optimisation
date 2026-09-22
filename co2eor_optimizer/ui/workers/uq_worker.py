import logging
from typing import Dict, Optional

from PyQt6.QtCore import QThread, pyqtSignal, QObject

from co2eor_optimizer.analysis.uq_engine import UncertaintyQuantificationEngine

logger = logging.getLogger(__name__)

class UQWorker(QThread):
    """Worker thread for running Uncertainty Quantification tasks."""
    progress_updated = pyqtSignal(int, str)
    result_ready = pyqtSignal(object)
    error_occurred = pyqtSignal(str)

    def __init__(self, engine: UncertaintyQuantificationEngine, method_name: str, kwargs: Dict, parent: Optional[QObject] = None):
        super().__init__(parent)
        self.engine = engine
        self.method_name = method_name
        self.kwargs = kwargs
        self._is_running = True

    def run(self):
        if not self._is_running: return
        method_title = self.method_name.replace('_', ' ').title()
        logger.info(f"UQWorker started for method: {self.method_name}")
        self.progress_updated.emit(0, f"Starting Uncertainty Quantification: {method_title}...")

        try:
            # TODO: The UQ engine should have a callback to report progress
            # self.engine.set_progress_callback(lambda p, m: self.progress_updated.emit(p, m))
            
            uq_func = getattr(self.engine, self.method_name)
            results = uq_func(**self.kwargs)
            
            if self._is_running:
                if results is not None:
                    self.progress_updated.emit(100, "UQ analysis complete.")
                    self.result_ready.emit(results)
                    logger.info(f"UQ method {self.method_name} completed successfully.")
                else:
                    msg = f"UQ analysis '{self.method_name}' produced no results."
                    logger.warning(msg)
                    self.error_occurred.emit(msg)
            
        except AttributeError as ae:
            msg = f"UQ engine is missing method '{self.method_name}'."
            logger.error(f"UQ Worker AttributeError: {msg}", exc_info=True)
            if self._is_running: self.error_occurred.emit(f"Configuration Error: {msg}")
        except Exception as e:
            logger.error(f"Error during UQ analysis: {e}", exc_info=True)
            if self._is_running: self.error_occurred.emit(f"An unexpected error occurred: {e}")
        finally:
            if not self._is_running: logger.info(f"UQWorker was stopped.")
            # self.engine.set_progress_callback(None)

    def stop(self):
        logger.info(f"UQWorker stop requested.")
        self._is_running = False
        # if hasattr(self.engine, 'cancel'): self.engine.cancel()