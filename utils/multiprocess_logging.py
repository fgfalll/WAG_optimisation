"""
Queue-based logging for safe multiprocess operation.
Routes all log records through a queue to a single listener process.
"""

import logging
import logging.handlers
import multiprocessing
import os
import sys
import threading
from functools import partial
from pathlib import Path
from typing import Optional

_queue: Optional[multiprocessing.Queue] = None
_listener: Optional[logging.handlers.QueueListener] = None
_listener_lock = threading.Lock()


class SafeFileHandler(logging.FileHandler):
    """
    File handler that gracefully handles rotation errors on Windows.

    During multiprocessing operations, multiple processes may attempt to
    rotate the same log file, causing PermissionError on Windows due to
    file locking. This handler silently drops log records when file access
    fails, preventing application crashes.

    This is typically used as the handler in the QueueListener for the
    main process, which is the only process that should write to the log file.
    """

    def emit(self, record: logging.LogRecord) -> None:
        """
        Emit a log record, handling file access errors gracefully.

        Args:
            record: The log record to emit.
        """
        try:
            super().emit(record)
        except (PermissionError, OSError) as e:
            # Silently drop log records if file is locked or inaccessible
            # This prevents crashes during multiprocessing on Windows
            # The logs are lost, but the application continues to run
            pass
        except Exception:
            # For any other unexpected error, also silently fail
            # to prevent cascading failures in the logging system
            pass


def setup_queue_logging(
    log_file_path: Path,
    level: int = logging.DEBUG,
    format_str: Optional[str] = None
) -> multiprocessing.Queue:
    """
    Set up queue-based logging for multiprocess safety.

    Args:
        log_file_path: Path to the log file
        level: Logging level
        format_str: Optional custom format string

    Returns:
        The queue used for log records
    """
    global _queue, _listener

    with _listener_lock:
        if _listener is not None:
            return _queue

        _queue = multiprocessing.Queue(maxsize=10000)

        os.makedirs(log_file_path.parent, exist_ok=True)
        file_handler = SafeFileHandler(log_file_path, mode='a', encoding='utf-8')
        file_handler.setLevel(level)

        if format_str is None:
            format_str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

        formatter = logging.Formatter(format_str)
        file_handler.setFormatter(formatter)

        _listener = logging.handlers.QueueListener(
            _queue,
            file_handler,
            respect_handler_level=True
        )
        _listener.start()

        _configure_root_logger(_queue, level)

        return _queue


def _configure_root_logger(queue: multiprocessing.Queue, level: int) -> None:
    """Configure root logger with QueueHandler."""
    root_logger = logging.getLogger()

    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    queue_handler = logging.handlers.QueueHandler(queue)
    queue_handler.setLevel(level)
    root_logger.addHandler(queue_handler)
    root_logger.setLevel(level)


def get_log_queue() -> Optional[multiprocessing.Queue]:
    """Get the current log queue."""
    return _queue


def is_logging_initialized() -> bool:
    """Check if queue logging has been initialized."""
    global _listener
    return _listener is not None


def _reset_all_loggers() -> None:
    """
    Reset all loggers in the logging hierarchy to prevent file handler conflicts.

    This is critical for multiprocessing on Windows where file locks can cause
    PermissionError when multiple processes try to rotate the same log file.
    """
    root = logging.getLogger()

    # The logging.Manager stores all created loggers
    # We need to iterate through the manager's loggerDict
    if hasattr(root, 'manager') and hasattr(root.manager, 'loggerDict'):
        logger_dict = root.manager.loggerDict

        # Create a list of items first to avoid RuntimeError during iteration
        loggers_to_reset = []
        for name, logger_obj in logger_dict.items():
            if isinstance(logger_obj, logging.Logger):
                loggers_to_reset.append(logger_obj)

        # Reset each logger to remove any file handlers
        for logger in loggers_to_reset:
            # Remove all handlers and close them to release file locks
            for handler in logger.handlers[:]:
                try:
                    handler.close()
                except Exception:
                    pass  # Ignore errors during handler close
                logger.removeHandler(handler)

            # Ensure propagation is enabled so logs reach the root logger
            logger.propagate = True
            logger.handlers = []
            logger.setLevel(logging.NOTSET)


def _worker_initializer(queue: multiprocessing.Queue, level: int = logging.INFO) -> None:
    """
    Initializer function for worker processes.
    Configures the worker to use the shared logging queue.

    This completely resets the logging configuration in the worker process
    to prevent file handler conflicts on Windows during multiprocessing.

    Args:
        queue: The queue to send log records to.
        level: Logging level to set for the worker process.
    """
    global _queue
    _queue = queue

    # Completely reset the root logger
    root_logger = logging.getLogger()

    # Remove and close all existing handlers from root logger
    for handler in root_logger.handlers[:]:
        try:
            handler.close()  # Important: close handler to release file locks
        except Exception:
            pass  # Ignore errors during handler close
        root_logger.removeHandler(handler)

    # Reset all existing loggers in the hierarchy
    _reset_all_loggers()

    # Add only the queue handler for multiprocessing-safe logging
    if queue is not None:
        queue_handler = logging.handlers.QueueHandler(queue)
        queue_handler.setLevel(level)
        root_logger.addHandler(queue_handler)
        root_logger.setLevel(level)
    else:
        root_logger.addHandler(logging.NullHandler())
        root_logger.setLevel(level)


def get_worker_initializer() -> callable:
    """
    Create an initializer function for worker processes.

    Returns:
        A callable initializer function that sets up logging for workers.
        Uses functools.partial for compatibility with Windows multiprocessing (pickling).
    """
    queue = get_log_queue()
    # Capture the current logging level of the main process
    current_level = logging.getLogger().level
    
    # Use partial instead of a local function to ensure picklability
    return partial(_worker_initializer, queue, current_level)


def configure_worker_logging(queue: Optional[multiprocessing.Queue] = None, level: int = logging.INFO) -> None:
    """
    Configure logging for worker processes.
    Worker processes should call this before any logging.

    This completely resets the logging configuration in the worker process
    to prevent file handler conflicts on Windows during multiprocessing.

    Args:
        queue: The queue to send log records to. If None, uses the global queue.
        level: Logging level to set for the worker process.
    """
    global _queue

    # Completely reset the root logger
    root_logger = logging.getLogger()

    # Remove and close all existing handlers from root logger
    for handler in root_logger.handlers[:]:
        try:
            handler.close()  # Important: close handler to release file locks
        except Exception:
            pass  # Ignore errors during handler close
        root_logger.removeHandler(handler)

    # Reset all existing loggers in the hierarchy
    _reset_all_loggers()

    log_queue = queue if queue is not None else _queue

    if log_queue is not None:
        queue_handler = logging.handlers.QueueHandler(log_queue)
        queue_handler.setLevel(level)
        root_logger.addHandler(queue_handler)
        root_logger.setLevel(level)
    else:
        root_logger.addHandler(logging.NullHandler())
        root_logger.setLevel(level)


def shutdown_queue_logging() -> None:
    """
    Gracefully shutdown the queue listener.
    """
    global _queue, _listener

    with _listener_lock:
        if _listener is not None:
            _listener.stop()
            _listener = None

        if _queue is not None:
            _queue.close()
            _queue.join_thread()
            _queue = None
