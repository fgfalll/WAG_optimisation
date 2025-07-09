import logging
from typing import Any, Dict, Optional

from PyQt6.QtCore import QThread, pyqtSignal, QObject

from co2eor_optimizer.core.optimisation_engine import OptimizationEngine

logger = logging.getLogger(__name__)

class OptimizationWorker(QThread):
    """Worker thread for running EOR optimization tasks with enhanced feedback."""
    progress_updated = pyqtSignal(str)
    result_ready = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)
    ga_progress_updated = pyqtSignal(dict)
    
    # --- [NEW] Signal specifically for when a target is not met ---
    target_unreachable = pyqtSignal(dict)

    def __init__(self, engine: OptimizationEngine, method_name: str, kwargs: Dict, parent: Optional[QObject] = None):
        super().__init__(parent)
        self.engine = engine
        self.method_name = method_name
        self.kwargs = kwargs
        self._is_running = True
        self._was_successful = False

    def was_successful(self) -> bool:
        """Returns True if the worker completed its task without being stopped or erroring."""
        return self._was_successful

    def _handle_ga_progress(self, progress_data: dict):
        """Emits the GA progress signal. This runs in the worker thread."""
        if self._is_running:
            self.ga_progress_updated.emit(progress_data)

    def run(self):
        if not self._is_running: return
        method_title = self.method_name.replace('_', ' ').title()
        logger.info(f"OptimizationWorker started for method: {self.method_name}")
        self.progress_updated.emit(f"Starting optimization: {method_title}...")

        try:
            # --- [ENHANCED] Inject callbacks for GA progress and cancellation ---
            if "genetic_algorithm" in self.method_name or "hybrid" in self.method_name:
                self.kwargs['progress_callback'] = self._handle_ga_progress
            
            # Provide a callable for the engine to check if it should keep running
            self.kwargs['worker_is_running_check'] = lambda: self._is_running
            
            # --- [MODIFIED] The engine now performs the full run and determines outcome ---
            # The engine is now responsible for deciding if the target was met.
            # We will pass a flag to tell it we want the special target-miss handling.
            self.kwargs['handle_target_miss'] = True
            
            optimization_func = getattr(self.engine, self.method_name)
            results = optimization_func(**self.kwargs)
            
            if self._is_running:
                # --- [REFACTORED] Logic to check for target failure is now cleaner ---
                # The engine now adds a flag to the results dictionary.
                if results.get('target_was_unreachable', False):
                    logger.warning(f"Target miss detected. Emitting target_unreachable.")
                    self.target_unreachable.emit(results)
                else:
                    logger.info("Target successfully met or standard optimization. Emitting result_ready.")
                    self.result_ready.emit(results)

                self._was_successful = True
                self.progress_updated.emit("Optimization complete.")
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

    def stop(self):
        logger.info(f"OptimizationWorker stop requested.")
        self._is_running = False