import logging
from typing import Dict, Optional
import numpy as np

from PyQt6.QtCore import QThread, pyqtSignal, QObject

from co2eor_optimizer.core.data_models import WellData, PVTProperties
from co2eor_optimizer.analysis.well_analysis import WellAnalysis

logger = logging.getLogger(__name__)

class WellAnalysisWorker(QThread):
    """Worker thread for calculating MMP profiles with interactive feedback."""
    # --- [MODIFIED] Simplified progress signal to a single string for the status bar ---
    progress_updated = pyqtSignal(str)
    result_ready = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)

    def __init__(self,
                 well_data: WellData,
                 pvt_data: PVTProperties,
                 parent: Optional[QObject] = None,
                 **kwargs):
        """
        Initializes the worker for a well analysis task.
        
        Args:
            well_data: The well data object.
            pvt_data: The PVT properties object.
            parent: The parent QObject.
            **kwargs: Can contain 'method', 'gas_composition', and 'c7_plus_mw_override'.
        """
        super().__init__(parent)
        self.well_data = well_data
        self.pvt_data = pvt_data
        
        self.method = kwargs.get('method', 'auto')
        self.gas_composition = kwargs.get('gas_composition')
        self.c7_plus_mw_override = kwargs.get('c7_plus_mw_override')
        
        self._is_running = True

    def run(self):
        if not self._is_running: return
        well_name = self.well_data.name
        logger.info(f"WellAnalysisWorker started for well '{well_name}' using method '{self.method}'.")
        self.progress_updated.emit(f"Starting MMP calculation for well {well_name}...")

        try:
            if not self.pvt_data:
                raise ValueError("PVT data is required for MMP calculation but was not provided.")

            analysis = WellAnalysis(self.well_data, self.pvt_data)
            
            # --- [ENHANCED] Define callback functions to pass to the analysis module ---
            def progress_handler(current_step, total_steps):
                if not self._is_running: return # Check flag before emitting
                percentage = int((current_step / total_steps) * 100)
                self.progress_updated.emit(f"Processing... ({percentage}%)")

            def stop_handler():
                return not self._is_running

            self.progress_updated.emit("Calculating MMP profile...")
            
            results = analysis.calculate_mmp_profile(
                method=self.method,
                gas_composition=self.gas_composition,
                c7_plus_mw_override=self.c7_plus_mw_override,
                progress_callback=progress_handler,
                is_stop_requested=stop_handler
            )
            
            if not self._is_running:
                logger.info(f"WellAnalysisWorker for '{well_name}' was stopped during calculation.")
                self.progress_updated.emit("Calculation cancelled.")
                return

            self.progress_updated.emit("Finalizing results...")
            
            processed_results = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in results.items()}

            self.progress_updated.emit("Analysis complete.")
            self.result_ready.emit(processed_results)
            logger.info(f"Well analysis for '{well_name}' completed successfully.")

        except Exception as e:
            logger.error(f"Error during well analysis for '{well_name}': {e}", exc_info=True)
            if self._is_running:
                self.error_occurred.emit(f"Analysis failed for well '{well_name}': {e}")
        finally:
            if not self._is_running:
                logger.info(f"WellAnalysisWorker for '{well_name}' was stopped.")

    def stop(self):
        """Requests the worker to stop processing."""
        logger.info(f"Stopping analysis for well '{self.well_data.name}'.")
        self._is_running = False