import logging
from typing import Optional, Any

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QTextBrowser, QPushButton, QHBoxLayout,
    QComboBox, QLabel, QMessageBox, QFileDialog
)
from PyQt6.QtCore import Qt, pyqtSignal, QObject
from PyQt6.QtGui import QColor, QTextCharFormat, QFont, QIcon

logger = logging.getLogger(__name__)

class QtLogSignal(QObject):
    """Defines a signal to carry log messages from any thread."""
    log_message_received = pyqtSignal(str, int)

class QtLogHandler(logging.Handler):
    """A logging handler that emits a Qt signal for each log record."""
    def __init__(self):
        super().__init__()
        self.emitter = QtLogSignal()
        self.setFormatter(logging.Formatter('%(asctime)s - %(levelname)-8s - %(name)s - %(message)s'))

    def emit(self, record: logging.LogRecord):
        msg = self.format(record)
        self.emitter.log_message_received.emit(msg, record.levelno)

class LoggingWidget(QWidget):
    """A widget that displays application logs with filtering and management controls."""
    LOG_LEVEL_COLORS = {
        logging.DEBUG: QColor("gray"),
        logging.INFO: QColor("black"),
        logging.WARNING: QColor("#E67E22"),
        logging.ERROR: QColor("#E74C3C"),
        logging.CRITICAL: QColor("#C0392B"),
    }
    LOG_LEVEL_MAP = {
        "ALL": logging.NOTSET, "DEBUG": logging.DEBUG, "INFO": logging.INFO,
        "WARNING": logging.WARNING, "ERROR": logging.ERROR, "CRITICAL": logging.CRITICAL,
    }

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.current_log_level_filter = logging.INFO
        self._qt_log_handler: Optional[QtLogHandler] = None
        self._setup_ui()
        self.setup_logging_capture()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)

        controls_layout = QHBoxLayout()
        controls_layout.addWidget(QLabel("Filter Level:"))
        self.level_filter_combo = QComboBox()
        for name, level in self.LOG_LEVEL_MAP.items():
            self.level_filter_combo.addItem(name, userData=level)
        self.level_filter_combo.setCurrentIndex(self.level_filter_combo.findData(logging.INFO))
        controls_layout.addWidget(self.level_filter_combo)
        controls_layout.addStretch(1)
        self.clear_button = QPushButton(QIcon.fromTheme("edit-clear"), " Clear")
        self.save_button = QPushButton(QIcon.fromTheme("document-save"), " Save")
        controls_layout.addWidget(self.clear_button)
        controls_layout.addWidget(self.save_button)
        layout.addLayout(controls_layout)

        self.log_browser = QTextBrowser()
        self.log_browser.setReadOnly(True)
        self.log_browser.setFont(QFont("Monospace", 9))
        layout.addWidget(self.log_browser)

        self.level_filter_combo.currentIndexChanged.connect(self._on_filter_level_changed)
        self.clear_button.clicked.connect(self.log_browser.clear)
        self.save_button.clicked.connect(self._save_logs_to_file)

    def setup_logging_capture(self):
        """Creates and adds the QtLogHandler to the root logger."""
        if self._qt_log_handler: return
        self._qt_log_handler = QtLogHandler()
        self._qt_log_handler.emitter.log_message_received.connect(self.append_log_message)
        logging.getLogger().addHandler(self._qt_log_handler)
        self._qt_log_handler.setLevel(logging.DEBUG) # Capture all levels, filter on display
        logger.info("LoggingWidget: Attached QtLogHandler to root logger.")

    def append_log_message(self, message: str, levelno: int):
        if levelno < self.current_log_level_filter: return

        cursor = self.log_browser.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        self.log_browser.setTextCursor(cursor)

        char_format = QTextCharFormat()
        char_format.setForeground(self.LOG_LEVEL_COLORS.get(levelno, QColor("black")))
        if levelno >= logging.ERROR:
            char_format.setFontWeight(QFont.Weight.Bold)
        
        cursor.insertText(message + '\n', char_format)

        scrollbar = self.log_browser.verticalScrollBar()
        if scrollbar.value() == scrollbar.maximum():
            scrollbar.setValue(scrollbar.maximum())

    def _on_filter_level_changed(self, index: int):
        level = self.level_filter_combo.itemData(index)
        if level is not None:
            self.current_log_level_filter = level
            level_name = logging.getLevelName(level)
            logger.info(f"Log display filter level changed to: {level_name}")
            self.log_browser.append(f"<i>--- Log filter set to {level_name} ---</i>")

    def _save_logs_to_file(self):
        filepath, _ = QFileDialog.getSaveFileName(self, "Save Logs", "app_logs.txt", "Text Files (*.txt);;All Files (*)")
        if not filepath: return
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(self.log_browser.toPlainText())
            QMessageBox.information(self, "Logs Saved", f"Logs successfully saved to:\n{filepath}")
        except Exception as e:
            logger.error(f"Error saving UI logs: {e}", exc_info=True)
            QMessageBox.critical(self, "Save Error", f"Could not save logs: {e}")

    def closeEvent(self, event: Any):
        """Removes the custom log handler when the widget is closed."""
        if self._qt_log_handler:
            logging.getLogger().removeHandler(self._qt_log_handler)
            self._qt_log_handler = None
            logger.info("LoggingWidget: Detached QtLogHandler from root logger.")
        super().closeEvent(event)