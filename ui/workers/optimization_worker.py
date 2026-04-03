import logging
from typing import Any, Dict, Optional

from PyQt6.QtCore import QThread, pyqtSignal, QObject
from core.optimisation_engine import OptimizationEngine

logger = logging.getLogger(__name__)

class OptimizationWorker(QThread):
    progress_updated = pyqtSignal(str)
    result_ready = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)
    ga_progress_updated = pyqtSignal(dict)
    target_unreachable = pyqtSignal(dict)

    def __init__(self, engine: OptimizationEngine, method_name: str, kwargs: Dict, parent: Optional[QObject] = None):
        super().__init__(parent)
        self.engine = engine
        self.method_name = method_name
        self.kwargs = kwargs
        self._is_running = True
        self._was_successful = False

    def was_successful(self) -> bool:
        return self._was_successful

    def run(self):
        if not self._is_running: return
        
        method_title = self.method_name.replace('_', ' ').title()
        logger.info(f"Worker started for method: {self.method_name} with kwargs: {list(self.kwargs.keys())}")
        self.progress_updated.emit(f"Starting: {method_title}...")

        try:
            # Pass callbacks and state-checkers into the kwargs for the engine to use
            # Set appropriate progress callback based on method type
            if 'genetic' in self.method_name or 'hybrid' in self.method_name:
                self.kwargs['convergence_progress_updated'] = self.ga_progress_updated.emit
            self.kwargs['worker_is_running_check'] = lambda: self._is_running
            self.kwargs['text_progress_callback'] = self.progress_updated.emit
            self.kwargs['handle_target_miss'] = True

            optimization_func = getattr(self.engine, self.method_name)
            results = optimization_func(**self.kwargs)
            
            if self._is_running:
                if results.get('target_was_unreachable', False):
                    self.target_unreachable.emit(results)
                else:
                    self.result_ready.emit(results)
                self._was_successful = True
                logger.info(f"Method {self.method_name} completed successfully.")

        except Exception as e:
            logger.error(f"Error during optimization in worker for method '{self.method_name}': {e}", exc_info=True)
            if self._is_running: self.error_occurred.emit(f"An unexpected error occurred: {e}")
        finally:
            if not self._was_successful:
                logger.warning(f"Worker for method '{self.method_name}' finished without success.")
            self.progress_updated.emit("Optimization run finished.")

    def stop(self):
        logger.info("Worker stop requested.")
        self._is_running = False