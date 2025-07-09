import logging
from typing import Dict, Optional
import pandas as pd

from PyQt6.QtCore import QThread, pyqtSignal, QObject

from co2eor_optimizer.analysis.sensitivity_analyzer import SensitivityAnalyzer

logger = logging.getLogger(__name__)

class SensitivityAnalysisWorker(QThread):
    """Worker thread for running sensitivity analysis tasks."""
    progress_updated = pyqtSignal(int, str)
    result_ready = pyqtSignal(pd.DataFrame)
    error_occurred = pyqtSignal(str)

    def __init__(self, analyzer: SensitivityAnalyzer, method_name: str, kwargs: Dict, parent: Optional[QObject] = None):
        super().__init__(parent)
        self.analyzer = analyzer
        self.method_name = method_name
        self.kwargs = kwargs
        self._is_running = True

    def run(self):
        if not self._is_running: return
        method_title = self.method_name.replace('_', ' ').title()
        logger.info(f"SensitivityAnalysisWorker started for method: {self.method_name}")
        self.progress_updated.emit(0, f"Starting Sensitivity Analysis: {method_title}...")

        try:
            # TODO: The analyzer should have a callback to report progress
            # self.analyzer.set_progress_callback(lambda p, m: self.progress_updated.emit(p, m))
            
            analysis_func = getattr(self.analyzer, self.method_name)
            results_df: pd.DataFrame = analysis_func(**self.kwargs)
            
            if self._is_running:
                self.progress_updated.emit(100, "Analysis complete.")
                self.result_ready.emit(results_df)
                logger.info(f"SA method {self.method_name} completed.")

        except AttributeError as ae:
            msg = f"Sensitivity analyzer is missing method '{self.method_name}'."
            logger.error(f"SA Worker AttributeError: {msg}", exc_info=True)
            if self._is_running: self.error_occurred.emit(f"Configuration Error: {msg}")
        except Exception as e:
            logger.error(f"Error during sensitivity analysis: {e}", exc_info=True)
            if self._is_running: self.error_occurred.emit(f"An unexpected error occurred: {e}")
        finally:
            if not self._is_running: logger.info(f"SensitivityAnalysisWorker was stopped.")
            # self.analyzer.set_progress_callback(None)

    def stop(self):
        logger.info(f"SensitivityAnalysisWorker stop requested.")
        self._is_running = False
        # if hasattr(self.analyzer, 'cancel'): self.analyzer.cancel()