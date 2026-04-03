"""
Centralized Error Manager for CO2 EOR Optimizer
Provides comprehensive error handling with proper propagation, logging, and user notification
"""

import logging
import traceback
import sys
from datetime import datetime
from typing import Optional, Dict, Any, Union, Callable

# Define ExceptionInfo for compatibility
try:
    from typing import ExceptionInfo
except ImportError:
    # For Python 3.11+ where ExceptionInfo was removed
    ExceptionInfo = Any
from dataclasses import dataclass, field
from enum import Enum
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QTextEdit,
    QGroupBox, QScrollArea, QSplitter, QFrame, QMessageBox
)
from PyQt6.QtCore import (
    Qt, QEvent, pyqtSignal, QTimer, QObject, QThread, pyqtSlot
)
from PyQt6.QtGui import QFont, QTextCursor, QPixmap, QIcon

logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels for classification"""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Categories of errors for better organization"""
    SYSTEM = "system"
    CALCULATION = "calculation"
    DATA = "data"
    UI = "ui"
    NETWORK = "network"
    FILE_IO = "file_io"
    CONFIGURATION = "configuration"
    OPTIMIZATION = "optimization"
    VALIDATION = "validation"
    UNKNOWN = "unknown"


@dataclass
class ErrorRecord:
    """Record of an error with full context"""
    timestamp: datetime
    severity: ErrorSeverity
    category: ErrorCategory
    title: str
    message: str
    exception: Optional[Exception] = None
    traceback_str: str = ""
    context: Dict[str, Any] = field(default_factory=dict)
    user_action_suggested: str = ""
    error_id: str = ""

    def __post_init__(self):
        """Generate error ID if not provided"""
        if not self.error_id:
            self.error_id = f"{self.severity.value}_{int(self.timestamp.timestamp()) % 100000}"


class UserErrorDialog(QMessageBox):
    """
    Enhanced error dialog with expandable details and copy functionality
    """

    def __init__(self, error_record: ErrorRecord, parent=None):
        super().__init__(parent)
        self.error_record = error_record
        self.setup_dialog()
        self.details_visible = False

    def setup_dialog(self):
        """Setup the error dialog with appropriate severity styling"""
        # Set dialog properties based on severity
        if self.error_record.severity == ErrorSeverity.CRITICAL:
            self.setIcon(QMessageBox.Icon.Critical)
            self.setWindowTitle("Critical Error")
        elif self.error_record.severity == ErrorSeverity.ERROR:
            self.setIcon(QMessageBox.Icon.Critical)
            self.setWindowTitle("Error")
        elif self.error_record.severity == ErrorSeverity.WARNING:
            self.setIcon(QMessageBox.Icon.Warning)
            self.setWindowTitle("Warning")
        else:
            self.setIcon(QMessageBox.Icon.Information)
            self.setWindowTitle("Information")

        # Set main error message
        self.setText(self.error_record.title)

        # Set informative text (summary)
        if self.error_record.user_action_suggested:
            self.setInformativeText(
                f"{self.error_record.message}\n\n"
                f"Suggested action: {self.error_record.user_action_suggested}"
            )
        else:
            self.setInformativeText(self.error_record.message)

        # Add custom buttons
        self.setStandardButtons(
            QMessageBox.StandardButton.Ok |
            QMessageBox.StandardButton.Retry |
            QMessageBox.StandardButton.Ignore
        )
        self.setDefaultButton(QMessageBox.StandardButton.Ok)

        # Add "Show Details" button
        self.details_button = QPushButton("Show Details")
        self.details_button.clicked.connect(self.toggle_details)
        self.addButton(self.details_button, QMessageBox.ButtonRole.ActionRole)

        # Add "Copy Traceback" button if there's a traceback
        if self.error_record.traceback_str:
            self.copy_button = QPushButton("Copy Traceback")
            self.copy_button.clicked.connect(self.copy_traceback)
            self.addButton(self.copy_button, QMessageBox.ButtonRole.ActionRole)

        # Store original size for animation
        self.original_size = None

    def toggle_details(self):
        """Toggle the visibility of detailed error information"""
        if not self.details_visible:
            self.show_details()
        else:
            self.hide_details()

    def show_details(self):
        """Expand dialog to show full error details"""
        self.details_visible = True
        self.details_button.setText("Hide Details")

        # Create details text
        details_text = self.create_details_text()

        # Set detailed text
        self.setDetailedText(details_text)

    def hide_details(self):
        """Collapse dialog to hide detailed error information"""
        self.details_visible = False
        self.details_button.setText("Show Details")
        self.setDetailedText("")

    def create_details_text(self) -> str:
        """Create comprehensive details text for the error"""
        details = []
        details.append(f"Error ID: {self.error_record.error_id}")
        details.append(f"Timestamp: {self.error_record.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        details.append(f"Category: {self.error_record.category.value}")
        details.append(f"Severity: {self.error_record.severity.value}")
        details.append("")

        if self.error_record.context:
            details.append("Context:")
            for key, value in self.error_record.context.items():
                details.append(f"  {key}: {value}")
            details.append("")

        if self.error_record.traceback_str:
            details.append("Full Traceback:")
            details.append(self.error_record.traceback_str)

        return "\n".join(details)

    def copy_traceback(self):
        """Copy full error details to clipboard"""
        try:
            from PyQt6.QtWidgets import QApplication
            app = QApplication.instance()
            clipboard = app.clipboard()

            # Copy full details
            full_text = self.create_details_text()
            clipboard.setText(full_text)

            # Show confirmation
            self.copy_button.setText("Copied!")
            QTimer.singleShot(1000, lambda: self.copy_button.setText("Copy Traceback"))

            logger.info(f"Error details copied to clipboard: {self.error_record.error_id}")

        except Exception as e:
            logger.error(f"Failed to copy error details: {e}")


class CentralErrorManager(QObject):
    """
    Centralized error management system for the entire application
    """

    # Signals
    error_occurred = pyqtSignal(ErrorRecord)
    error_cleared = pyqtSignal(str)  # error_id
    error_count_changed = pyqtSignal(int)

    def __init__(self):
        super().__init__()
        self.error_records: list[ErrorRecord] = []
        self.error_callbacks: Dict[str, Callable] = {}
        self.max_error_history = 1000

        # Setup global exception handler
        self.setup_global_exception_handler()

    def setup_global_exception_handler(self):
        """Setup global exception handler to catch unhandled exceptions"""
        # Store original handlers
        self.original_excepthook = sys.excepthook

        # Set new exception handler
        sys.excepthook = self.handle_unhandled_exception

    def handle_unhandled_exception(self, exc_type, exc_value, exc_traceback):
        """Handle unhandled exceptions globally"""
        # Create error record
        error_record = self.create_error_record(
            title=f"Unhandled {exc_type.__name__}",
            message=str(exc_value),
            exception=exc_value,
            severity=ErrorSeverity.CRITICAL,
            category=ErrorCategory.SYSTEM
        )

        # Log the error
        logger.critical(f"Unhandled exception: {exc_type.__name__}: {exc_value}", exc_info=True)

        # Emit signal
        self.error_occurred.emit(error_record)

        # Call original handler to maintain default behavior
        if self.original_excepthook:
            self.original_excepthook(exc_type, exc_value, exc_traceback)

    def create_error_record(
        self,
        title: str,
        message: str,
        exception: Optional[Exception] = None,
        severity: ErrorSeverity = ErrorSeverity.ERROR,
        category: ErrorCategory = ErrorCategory.UNKNOWN,
        context: Optional[Dict[str, Any]] = None,
        user_action_suggested: str = ""
    ) -> ErrorRecord:
        """Create a standardized error record"""

        # Generate traceback if exception provided
        traceback_str = ""
        if exception:
            traceback_str = "".join(traceback.format_exception(
                type(exception), exception, exception.__traceback__
            ))

        return ErrorRecord(
            timestamp=datetime.now(),
            severity=severity,
            category=category,
            title=title,
            message=message,
            exception=exception,
            traceback_str=traceback_str,
            context=context or {},
            user_action_suggested=user_action_suggested
        )

    @pyqtSlot(ErrorRecord)
    def handle_error(self, error_record: ErrorRecord, show_dialog: bool = True):
        """Handle an error record"""
        # Add to history
        self.add_error_record(error_record)

        # Log based on severity
        if error_record.severity == ErrorSeverity.CRITICAL:
            logger.critical(f"{error_record.title}: {error_record.message}", exc_info=error_record.exception)
        elif error_record.severity == ErrorSeverity.ERROR:
            logger.error(f"{error_record.title}: {error_record.message}", exc_info=error_record.exception)
        elif error_record.severity == ErrorSeverity.WARNING:
            logger.warning(f"{error_record.title}: {error_record.message}")
        else:
            logger.info(f"{error_record.title}: {error_record.message}")

        # Emit signal
        self.error_occurred.emit(error_record)

        # Show dialog if requested
        if show_dialog and error_record.severity in [ErrorSeverity.ERROR, ErrorSeverity.CRITICAL]:
            # Show dialog in main thread
            QTimer.singleShot(0, lambda: self.show_error_dialog(error_record))

        # Call registered callbacks
        for callback_id, callback in self.error_callbacks.items():
            try:
                callback(error_record)
            except Exception as e:
                logger.error(f"Error in callback {callback_id}: {e}")

    def show_error_dialog(self, error_record: ErrorRecord):
        """Show error dialog in main thread"""
        try:
            dialog = UserErrorDialog(error_record)
            dialog.exec()
        except Exception as e:
            logger.error(f"Failed to show error dialog: {e}")

    def add_error_record(self, error_record: ErrorRecord):
        """Add error record to history"""
        self.error_records.append(error_record)

        # Trim history if too long
        if len(self.error_records) > self.max_error_history:
            self.error_records = self.error_records[-self.max_error_history:]

        # Emit count change
        self.error_count_changed.emit(len(self.error_records))

    def clear_error(self, error_id: str):
        """Clear specific error from history"""
        self.error_records = [e for e in self.error_records if e.error_id != error_id]
        self.error_cleared.emit(error_id)
        self.error_count_changed.emit(len(self.error_records))

    def clear_all_errors(self):
        """Clear all error records"""
        self.error_records.clear()
        self.error_count_changed.emit(0)

    def get_errors_by_severity(self, severity: ErrorSeverity) -> list[ErrorRecord]:
        """Get errors filtered by severity"""
        return [e for e in self.error_records if e.severity == severity]

    def get_errors_by_category(self, category: ErrorCategory) -> list[ErrorRecord]:
        """Get errors filtered by category"""
        return [e for e in self.error_records if e.category == category]

    def register_error_callback(self, callback_id: str, callback: Callable[[ErrorRecord], None]):
        """Register a callback to be called when errors occur"""
        self.error_callbacks[callback_id] = callback

    def unregister_error_callback(self, callback_id: str):
        """Unregister an error callback"""
        if callback_id in self.error_callbacks:
            del self.error_callbacks[callback_id]


class ErrorDisplayWidget(QWidget):
    """
    Comprehensive error display widget with filtering and detailed view
    """

    def __init__(self, error_manager: CentralErrorManager, parent=None):
        super().__init__(parent)
        self.error_manager = error_manager
        self.current_filter = None
        self.setup_ui()
        self.connect_signals()

    def setup_ui(self):
        """Setup the UI components"""
        layout = QVBoxLayout(self)

        # Header with filters and controls
        header_frame = QFrame()
        header_layout = QHBoxLayout(header_frame)

        # Title
        title_label = QLabel("Error Monitor")
        title_label.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        header_layout.addWidget(title_label)

        header_layout.addStretch()

        # Filter buttons
        self.filter_buttons = {}
        for severity in ErrorSeverity:
            btn = QPushButton(severity.value.capitalize())
            btn.setCheckable(True)
            btn.clicked.connect(lambda checked, s=severity: self.toggle_filter(s))
            self.filter_buttons[severity] = btn
            header_layout.addWidget(btn)

        # Clear button
        clear_btn = QPushButton("Clear All")
        clear_btn.clicked.connect(self.clear_all_errors)
        header_layout.addWidget(clear_btn)

        layout.addWidget(header_frame)

        # Error list with details
        splitter = QSplitter(Qt.Orientation.Vertical)

        # Error list (top)
        self.error_list = QTextEdit()
        self.error_list.setReadOnly(True)
        self.error_list.setFont(QFont("monospace", 9))
        self.error_list.setMaximumHeight(200)
        splitter.addWidget(self.error_list)

        # Error details (bottom)
        details_group = QGroupBox("Error Details")
        details_layout = QVBoxLayout(details_group)

        self.details_text = QTextEdit()
        self.details_text.setReadOnly(True)
        self.details_text.setFont(QFont("monospace", 9))
        details_layout.addWidget(self.details_text)

        # Details control buttons
        details_buttons = QHBoxLayout()

        self.copy_details_btn = QPushButton("Copy Details")
        self.copy_details_btn.clicked.connect(self.copy_selected_error)
        details_buttons.addWidget(self.copy_details_btn)

        details_buttons.addStretch()

        details_layout.addLayout(details_buttons)
        splitter.addWidget(details_group)

        # Set splitter sizes
        splitter.setSizes([200, 400])
        layout.addWidget(splitter)

        # Status bar
        self.status_label = QLabel("No errors")
        layout.addWidget(self.status_label)

    def connect_signals(self):
        """Connect to error manager signals"""
        self.error_manager.error_occurred.connect(self.on_error_occurred)
        self.error_manager.error_count_changed.connect(self.update_error_count)

    @pyqtSlot(ErrorRecord)
    def on_error_occurred(self, error_record: ErrorRecord):
        """Handle new error occurrence"""
        self.add_error_to_list(error_record)
        self.update_status()

    def add_error_to_list(self, error_record: ErrorRecord):
        """Add error to the list display"""
        # Format error line
        time_str = error_record.timestamp.strftime("%H:%M:%S")
        error_line = f"[{time_str}] {error_record.severity.value.upper()}: {error_record.title}"

        # Apply color based on severity
        color = self.get_severity_color(error_record.severity)

        self.error_list.setTextColor(color)
        self.error_list.append(error_line)

        # Auto-scroll to latest error
        cursor = self.error_list.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self.error_list.setTextCursor(cursor)

    def get_severity_color(self, severity: ErrorSeverity):
        """Get color for error severity"""
        colors = {
            ErrorSeverity.DEBUG: Qt.GlobalColor.gray,
            ErrorSeverity.INFO: Qt.GlobalColor.blue,
            ErrorSeverity.WARNING: Qt.GlobalColor.darkYellow,
            ErrorSeverity.ERROR: Qt.GlobalColor.red,
            ErrorSeverity.CRITICAL: Qt.GlobalColor.darkRed
        }
        return colors.get(severity, Qt.GlobalColor.black)

    def update_error_count(self, count: int):
        """Update error count display"""
        if count == 0:
            self.status_label.setText("No errors")
        else:
            self.status_label.setText(f"{count} error(s) recorded")

    def update_status(self):
        """Update status based on current errors"""
        if self.error_manager.error_records:
            latest_error = self.error_manager.error_records[-1]
            self.status_label.setText(
                f"Latest: {latest_error.severity.value.upper()} - {latest_error.title}"
            )

    def toggle_filter(self, severity: ErrorSeverity):
        """Toggle error filter by severity"""
        button = self.filter_buttons[severity]
        button.setChecked(not button.isChecked())
        self.refresh_error_list()

    def refresh_error_list(self):
        """Refresh the error list based on current filters"""
        self.error_list.clear()

        # Get active filters
        active_severities = [
            s for s, btn in self.filter_buttons.items()
            if btn.isChecked()
        ]

        if not active_severities:
            active_severities = list(ErrorSeverity)

        # Filter and display errors
        for error_record in self.error_manager.error_records:
            if error_record.severity in active_severities:
                self.add_error_to_list(error_record)

    def clear_all_errors(self):
        """Clear all errors"""
        self.error_manager.clear_all_errors()
        self.error_list.clear()
        self.details_text.clear()

    def copy_selected_error(self):
        """Copy selected error details to clipboard"""
        try:
            from PyQt6.QtWidgets import QApplication
            app = QApplication.instance()
            clipboard = app.clipboard()

            if self.details_text.toPlainText():
                clipboard.setText(self.details_text.toPlainText())
                self.copy_details_btn.setText("Copied!")
                QTimer.singleShot(1000, lambda: self.copy_details_btn.setText("Copy Details"))
                logger.info("Error details copied to clipboard")
            else:
                logger.warning("No error details to copy")

        except Exception as e:
            logger.error(f"Failed to copy error details: {e}")


# Global error manager instance
_error_manager = None


def get_error_manager() -> CentralErrorManager:
    """Get the global error manager instance"""
    global _error_manager
    if _error_manager is None:
        _error_manager = CentralErrorManager()
    return _error_manager


def handle_error(
    title: str,
    message: str,
    exception: Optional[Exception] = None,
    severity: ErrorSeverity = ErrorSeverity.ERROR,
    category: ErrorCategory = ErrorCategory.UNKNOWN,
    context: Optional[Dict[str, Any]] = None,
    user_action_suggested: str = "",
    show_dialog: bool = True
) -> ErrorRecord:
    """
    Convenience function to handle errors throughout the application

    Args:
        title: Brief error title
        message: Detailed error message
        exception: The exception that occurred (if any)
        severity: Error severity level
        category: Error category
        context: Additional context information
        user_action_suggested: Suggested action for user
        show_dialog: Whether to show error dialog

    Returns:
        ErrorRecord: The created error record
    """
    error_manager = get_error_manager()
    error_record = error_manager.create_error_record(
        title=title,
        message=message,
        exception=exception,
        severity=severity,
        category=category,
        context=context,
        user_action_suggested=user_action_suggested
    )

    error_manager.handle_error(error_record, show_dialog)
    return error_record


def handle_caught_exception(
    operation: str,
    exception: Exception,
    context: Optional[Dict[str, Any]] = None,
    user_action_suggested: str = "",
    show_dialog: bool = True,
    severity: ErrorSeverity = ErrorSeverity.ERROR,
    category: ErrorCategory = ErrorCategory.UNKNOWN
) -> ErrorRecord:
    """
    Convenience function to handle caught exceptions with proper context

    Args:
        operation: Description of the operation that failed
        exception: The caught exception
        context: Additional context information
        user_action_suggested: Suggested action for user
        show_dialog: Whether to show error dialog
        severity: Error severity level
        category: Error category

    Returns:
        ErrorRecord: The created error record
    """
    return handle_error(
        title=f"Failed to {operation}",
        message=f"{operation} failed: {str(exception)}",
        exception=exception,
        severity=severity,
        category=category,
        context=context,
        user_action_suggested=user_action_suggested,
        show_dialog=show_dialog
    )


# Exception context manager for automatic error handling
class ErrorHandler:
    """Context manager for automatic error handling"""

    def __init__(
        self,
        operation: str,
        context: Optional[Dict[str, Any]] = None,
        user_action_suggested: str = "",
        show_dialog: bool = True,
        severity: ErrorSeverity = ErrorSeverity.ERROR,
        category: ErrorCategory = ErrorCategory.UNKNOWN,
        reraise: bool = False
    ):
        self.operation = operation
        self.context = context
        self.user_action_suggested = user_action_suggested
        self.show_dialog = show_dialog
        self.severity = severity
        self.category = category
        self.reraise = reraise

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            # Handle the exception
            handle_caught_exception(
                operation=self.operation,
                exception=exc_val,
                context=self.context,
                user_action_suggested=self.user_action_suggested,
                show_dialog=self.show_dialog,
                severity=self.severity,
                category=self.category
            )

            # Re-raise if requested
            if self.reraise:
                return False  # Re-raise
            else:
                return True  # Suppress exception
        return False


# Decorator for function error handling
def error_handler(
    operation: str = "",
    context: Optional[Dict[str, Any]] = None,
    user_action_suggested: str = "",
    show_dialog: bool = True,
    severity: ErrorSeverity = ErrorSeverity.ERROR,
    category: ErrorCategory = ErrorCategory.UNKNOWN,
    reraise: bool = False
):
    """
    Decorator for automatic function error handling

    Args:
        operation: Description of the operation
        context: Additional context information
        user_action_suggested: Suggested action for user
        show_dialog: Whether to show error dialog
        severity: Error severity level
        category: Error category
        reraise: Whether to re-raise the exception
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            op_name = operation or f"execute {func.__name__}"
            try:
                return func(*args, **kwargs)
            except Exception as e:
                handle_caught_exception(
                    operation=op_name,
                    exception=e,
                    context=context,
                    user_action_suggested=user_action_suggested,
                    show_dialog=show_dialog,
                    severity=severity,
                    category=category
                )
                if reraise:
                    raise
                return None
        return wrapper
    return decorator


# Initialize the error manager when this module is imported
_error_manager = get_error_manager()
logger.info("Central error manager initialized successfully")


"""
Global Error Handler Integration Module
Provides easy access to the centralized error management system from anywhere in the project
"""

from typing import Optional, Dict, Any, Callable, Union

# Define ExceptionInfo type for compatibility
try:
    from typing import ExceptionInfo
except ImportError:
    # For Python 3.11+ where ExceptionInfo was removed
    from typing import Any
    ExceptionInfo = Any


# Convenience functions for the entire project
def report_error(
    title: str,
    message: str,
    exception: Optional[Exception] = None,
    severity: str = ErrorSeverity.ERROR,
    category: str = ErrorCategory.UNKNOWN,
    context: Optional[Dict[str, Any]] = None,
    user_action_suggested: str = "",
    show_dialog: bool = True
) -> Optional[ErrorRecord]:
    """
    Report an error from anywhere in the project

    Args:
        title: Brief error title
        message: Detailed error message
        exception: The exception that occurred (if any)
        severity: Error severity level (use ErrorSeverity constants)
        category: Error category (use ErrorCategory constants)
        context: Additional context information
        user_action_suggested: Suggested action for user
        show_dialog: Whether to show error dialog

    Returns:
        ErrorRecord if available, None otherwise
    """
    try:
        return handle_error(
            title=title,
            message=message,
            exception=exception,
            severity=severity,
            category=category,
            context=context,
            user_action_suggested=user_action_suggested,
            show_dialog=show_dialog
        )
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"Failed to report error: {e}")
        logger.error(f"Original error: {title} - {message}")
        if exception:
            logger.error(f"Original exception: {exception}")
        return None


def report_caught_error(
    operation: str,
    exception: Exception,
    context: Optional[Dict[str, Any]] = None,
    user_action_suggested: str = "",
    show_dialog: bool = True,
    severity: str = ErrorSeverity.ERROR,
    category: str = ErrorCategory.UNKNOWN
) -> Optional[ErrorRecord]:
    """
    Report a caught exception from anywhere in the project

    Args:
        operation: Description of the operation that failed
        exception: The caught exception
        context: Additional context information
        user_action_suggested: Suggested action for user
        show_dialog: Whether to show error dialog
        severity: Error severity level
        category: Error category

    Returns:
        ErrorRecord if available, None otherwise
    """
    try:
        return handle_caught_exception(
            operation=operation,
            exception=exception,
            context=context,
            user_action_suggested=user_action_suggested,
            show_dialog=show_dialog,
            severity=severity,
            category=category
        )
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"Failed to report caught error: {e}")
        logger.error(f"Original operation: {operation}")
        logger.error(f"Original exception: {exception}")
        return None


def safe_execute(
    operation: str,
    context: Optional[Dict[str, Any]] = None,
    user_action_suggested: str = "",
    show_dialog: bool = True,
    severity: str = ErrorSeverity.ERROR,
    category: str = ErrorCategory.UNKNOWN,
    reraise: bool = False
):
    """
    Context manager for safe operation execution with automatic error handling

    Args:
        operation: Description of the operation
        context: Additional context information
        user_action_suggested: Suggested action for user
        show_dialog: Whether to show error dialog
        severity: Error severity level
        category: Error category
        reraise: Whether to re-raise the exception

    Returns:
        ErrorHandler context manager
    """
    try:
        return ErrorHandler(
            operation=operation,
            context=context,
            user_action_suggested=user_action_suggested,
            show_dialog=show_dialog,
            severity=severity,
            category=category,
            reraise=reraise
        )
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"Failed to create error handler: {e}")

        # Return fallback handler
        class FallbackErrorHandler:
            def __init__(self):
                self.operation = operation
                self.exception = None
            def __enter__(self):
                return self
            def __exit__(self, exc_type, exc_val, exc_tb):
                if exc_type is not None:
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.error(f"Error in {self.operation}: {exc_val}")
                    self.exception = exc_val
                return not reraise  # Suppress unless reraise is True

        return FallbackErrorHandler()


def safe_function(
    operation: str = "",
    context: Optional[Dict[str, Any]] = None,
    user_action_suggested: str = "",
    show_dialog: bool = True,
    severity: str = ErrorSeverity.ERROR,
    category: str = ErrorCategory.UNKNOWN,
    reraise: bool = False
):
    """
    Decorator for safe function execution with automatic error handling

    Args:
        operation: Description of the operation
        context: Additional context information
        user_action_suggested: Suggested action for user
        show_dialog: Whether to show error dialog
        severity: Error severity level
        category: Error category
        reraise: Whether to re-raise the exception

    Returns:
        Decorated function
    """
    try:
        return error_handler(
            operation=operation,
            context=context,
            user_action_suggested=user_action_suggested,
            show_dialog=show_dialog,
            severity=severity,
            category=category,
            reraise=reraise
        )
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"Failed to create function decorator: {e}")

        # Return fallback decorator
        def fallback_decorator(func):
            def wrapper(*args, **kwargs):
                try:
                    return func(*args, **kwargs)
                except Exception as exc:
                    import logging
                    logger = logging.getLogger(__name__)
                    op_name = operation or f"execute {func.__name__}"
                    logger.error(f"Error in {op_name}: {exc}")
                    if reraise:
                        raise
                    return None
            return wrapper
        return fallback_decorator


def get_error_manager_instance():
    """
    Get the global error manager instance

    Returns:
        CentralErrorManager instance or None if unavailable
    """
    try:
        return get_error_manager()
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"Failed to get error manager: {e}")
        return None
