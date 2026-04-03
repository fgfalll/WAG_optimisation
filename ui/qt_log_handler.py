import logging
from typing import Dict
from PyQt6.QtCore import QObject, pyqtSignal
from PyQt6.QtGui import QColor


class QtLogHandler(logging.Handler):
    class Emitter(QObject):
        log_record_received = pyqtSignal(logging.LogRecord)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.emitter = self.Emitter()

    def emit(self, record: logging.LogRecord):
        self.emitter.log_record_received.emit(record)


class LogFormatter:
    LEVEL_COLORS: Dict[int, QColor] = {
        logging.DEBUG: QColor("gray"),
        logging.INFO: QColor("black"),
        logging.WARNING: QColor("#E67E22"),
        logging.ERROR: QColor("#E74C3C"),
        logging.CRITICAL: QColor("#C0392B"),
    }
    DEFAULT_COLOR = QColor("black")

    def __init__(self):
        self.formatter = logging.Formatter(
            "%(asctime)s - %(levelname)-8s - %(name)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )

    def format(self, record: logging.LogRecord) -> str:
        color = self.LEVEL_COLORS.get(record.levelno, self.DEFAULT_COLOR)
        if record.exc_text:
            record.message += "\n" + record.exc_text
        message = self.formatter.format(record)
        message = self.escape_html(message)
        if record.levelno >= logging.ERROR:
            return f'<font color="{color.name()}"><b>{message}</b></font>'
        else:
            return f'<font color="{color.name()}">{message}</font>'

    @staticmethod
    def escape_html(text: str) -> str:
        return (
            text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&#39;")
        )
