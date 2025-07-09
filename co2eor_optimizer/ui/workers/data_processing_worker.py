import logging
from typing import List, Optional

from PyQt6.QtCore import QThread, pyqtSignal, QObject

from co2eor_optimizer.data_processor import DataProcessor

logger = logging.getLogger(__name__)


class DataProcessingWorker(QThread):
    """
    Worker thread to run the DataProcessor in the background, preventing UI freezes.
    This worker is enhanced to handle real-time progress updates from the processor.
    """
    # Signals to communicate with the main UI thread
    progress_updated = pyqtSignal(int, str)  # (percentage, message)
    result_ready = pyqtSignal(dict)          # (dictionary_with_results)
    error_occurred = pyqtSignal(str)         # (error_message)

    def __init__(self, filepaths: List[str], parent: Optional[QObject] = None):
        super().__init__(parent)
        self.filepaths = filepaths
        self._is_running = True

    def run(self):
        """The main execution method for the thread."""
        if not self._is_running:
            return
        
        logger.info(f"DataProcessingWorker started for {len(self.filepaths)} files.")
        self.progress_updated.emit(0, "Starting data processing...")

        try:
            # 1. Initialize the processor with the list of files
            processor = DataProcessor(filepaths=self.filepaths)
            self.progress_updated.emit(5, "Processor initialized. Parsing files...")

            # 2. Define the callback function that will emit progress signals
            # The processing phase will cover the 5% to 95% progress range.
            def report_progress(percentage: int, message: str):
                if self._is_running:
                    # Scale the processor's 0-100% progress to our 5-95% range
                    scaled_percentage = 5 + int(percentage * 0.90)
                    self.progress_updated.emit(scaled_percentage, message)

            # 3. Execute the processing, passing the callback
            processed_data = processor.process_files(progress_callback=report_progress)
            
            # Check if the process was stopped during the operation
            if not self._is_running:
                logger.info("DataProcessingWorker was stopped during file processing.")
                return

            self.progress_updated.emit(95, "Finalizing results...")

            # 4. Assemble the final results dictionary for the UI
            final_results = {
                "well_data": processed_data.get('well_data', []),
                "reservoir_data": processed_data.get('reservoir_data', None),
                "summary": processor.get_summary(),
                "failed_files": processed_data.get('failed_files', {})
            }

            if self._is_running:
                self.progress_updated.emit(100, "Processing complete.")
                self.result_ready.emit(final_results)
                logger.info("DataProcessingWorker completed successfully.")

        except Exception as e:
            logger.error(f"An unhandled error occurred in DataProcessingWorker: {e}", exc_info=True)
            if self._is_running:
                self.error_occurred.emit(f"A critical error occurred: {e}")
        finally:
            if not self._is_running:
                logger.info("DataProcessingWorker was cleanly stopped before completion.")

    def stop(self):
        """Requests the worker to stop its operation safely."""
        logger.info("Stop request received by DataProcessingWorker.")
        self._is_running = False
        self.terminate() # Forcibly end if needed, though clean exit is preferred
        self.wait(2000) # Wait up to 2 seconds for thread to finish