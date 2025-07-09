import logging
from typing import Any, Dict, Optional

from PyQt6.QtCore import QThread, pyqtSignal, QObject

from co2eor_optimizer.core.optimisation_engine import OptimizationEngine

logger = logging.getLogger(__name__)

class OptimizationWorker(QThread):
    """Worker thread for running EOR optimization tasks."""
    progress_updated = pyqtSignal(int, str)
    result_ready = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)

    def __init__(self, engine: OptimizationEngine, method_name: str, kwargs: Dict, parent: Optional[QObject] = None):
        super().__init__(parent)
        self.engine = engine
        self.method_name = method_name
        self.kwargs = kwargs
        self._is_running = True

    def run(self):
        if not self._is_running: return
        method_title = self.method_name.replace('_', ' ').title()
        logger.info(f"OptimizationWorker started for method: {self.method_name}")
        self.progress_updated.emit(0, f"Starting optimization: {method_title}...")

        try:
            # TODO: The engine should have a callback to report progress
            # self.engine.set_progress_callback(lambda p, m: self.progress_updated.emit(p, m))
            
            optimization_func = getattr(self.engine, self.method_name)
            results = optimization_func(**self.kwargs)
            
            if self._is_running:
                self.progress_updated.emit(100, "Optimization complete.")
                self.result_ready.emit(results)
                logger.info(f"Optimization method {self.method_name} completed successfully.")

        except AttributeError as ae:
            msg = f"Optimization engine is missing method '{self.method_name}'."
            logger.error(f"OptimizationWorker AttributeError: {msg}", exc_info=True)
            if self._is_running: self.error_occurred.emit(f"Configuration Error: {msg}")
        except Exception as e:
            logger.error(f"Error during optimization in worker: {e}", exc_info=True)
            if self._is_running: self.error_occurred.emit(f"An unexpected error occurred: {e}")
        finally:
            if not self._is_running: logger.info(f"OptimizationWorker was stopped.")
            # self.engine.set_progress_callback(None) # Unregister callback

    def stop(self):
        logger.info(f"OptimizationWorker stop requested.")
        self._is_running = False
        # If the engine supports cancellation, trigger it here.
        # if hasattr(self.engine, 'cancel'): self.engine.cancel()