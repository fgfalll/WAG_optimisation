import logging
from abc import abstractmethod
from typing import Optional

from PyQt6.QtCore import QThread, pyqtSignal

logger = logging.getLogger(__name__)


class BaseWorker(QThread):
    """Base class for worker threads with common stop/running functionality."""

    progress_updated = pyqtSignal(str)
    result_ready = pyqtSignal(object)
    error_occurred = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, parent: Optional[QThread] = None):
        super().__init__(parent)
        self._is_running = True

    def stop(self):
        logger.info(f"{self.__class__.__name__} stop requested.")
        self._is_running = False

    def was_stopped(self) -> bool:
        return not self._is_running

    @abstractmethod
    def _do_work(self):
        """Override this method to implement actual work."""
        pass

    def run(self):
        if not self._is_running:
            logger.info(f"{self.__class__.__name__} run aborted - was stopped before starting.")
            return

        try:
            self._do_work()
        except Exception as e:
            logger.error(f"Error in {self.__class__.__name__}: {e}", exc_info=True)
            if self._is_running:
                self.error_occurred.emit(f"An unexpected error occurred: {e}")
        finally:
            self.finished.emit()
            self._is_running = True