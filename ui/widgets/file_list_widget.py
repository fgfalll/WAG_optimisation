import logging
from typing import List, Optional, Union
from pathlib import Path

from PyQt6.QtWidgets import (
    QListWidget, QListWidgetItem, QAbstractItemView, QMenu, QMessageBox,
    QSizePolicy, QWidget
)
from PyQt6.QtGui import QIcon, QAction, QDesktopServices, QColor
from PyQt6.QtCore import Qt, QUrl, pyqtSignal, QEvent

logger = logging.getLogger(__name__)

class FileListViewWidget(QListWidget):
    """
    A custom QListWidget to display a list of files with status icons,
    tooltips, and a context menu for actions.
    """
    file_removed = pyqtSignal(str)
    files_cleared = pyqtSignal()
    file_double_clicked = pyqtSignal(str)

    # --- Status Constants ---
    STATUS_OK = "ok"
    STATUS_WARNING = "warning"
    STATUS_ERROR = "error"
    STATUS_PROCESSING = "processing"
    STATUS_PENDING = "pending"

    # --- UI Configuration (translatable strings removed) ---
    STATUS_INFO = {
        STATUS_OK: {"icon_path": "resources/icons/status-ok.png", "color": QColor("#1E8449")},
        STATUS_WARNING: {"icon_path": "resources/icons/status-warning.png", "color": QColor("#F39C12")},
        STATUS_ERROR: {"icon_path": "resources/icons/status-error.png", "color": QColor("#C0392B")},
        STATUS_PROCESSING: {"icon_path": "resources/icons/status-processing.png", "color": QColor("#2980B9")},
        STATUS_PENDING: {"icon_path": "resources/icons/status-pending.png", "color": QColor("#808080")},
    }

    # --- Custom Data Roles for Storing State ---
    StatusRole = Qt.ItemDataRole.UserRole + 1
    StatusMessageRole = Qt.ItemDataRole.UserRole + 2


    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.setAlternatingRowColors(True)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._show_context_menu)
        self.itemDoubleClicked.connect(self._handle_item_double_click)

        self.setStyleSheet("""
            QListWidget { font-size: 10pt; }
            QListWidget::item { padding: 4px; }
            QListWidget::item:selected { background-color: #d4e8ff; color: black; }
        """)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.setMinimumHeight(100)

    def changeEvent(self, event: QEvent):
        """Handle language change events to re-translate the UI."""
        if event.type() == QEvent.Type.LanguageChange:
            self.retranslateUi()
        super().changeEvent(event)

    def retranslateUi(self):
        """Update tooltips for all existing items to reflect the current language."""
        for i in range(self.count()):
            item = self.item(i)
            if item:
                item.setToolTip(self._generate_tooltip_text(item))

    def _get_translated_status_tooltip(self, status_key: str) -> str:
        """Returns the translated tooltip text for a given status key."""
        tooltips = {
            self.STATUS_OK: self.tr("Parsed successfully"),
            self.STATUS_WARNING: self.tr("Warning during processing"),
            self.STATUS_ERROR: self.tr("Error during processing"),
            self.STATUS_PROCESSING: self.tr("Currently processing..."),
            self.STATUS_PENDING: self.tr("Pending processing"),
        }
        return tooltips.get(status_key, self.tr("File"))

    def _generate_tooltip_text(self, item: QListWidgetItem) -> str:
        """Generates a fully translated tooltip for a list item using its stored data."""
        full_path = item.data(Qt.ItemDataRole.UserRole)
        status = item.data(self.StatusRole)
        status_message = item.data(self.StatusMessageRole)

        if not all((full_path, status is not None)):
            return ""

        base_tooltip = self._get_translated_status_tooltip(status)
        
        tooltip_text = (
            f"{self.tr('File')}: {Path(full_path).name}\n"
            f"{self.tr('Path')}: {full_path}\n"
            f"{self.tr('Status')}: {base_tooltip}"
        )
        if status_message:
            tooltip_text += f"\n\n{self.tr('Details')}:\n{status_message}"
        
        return tooltip_text

    def _get_status_icon(self, status_key: str) -> QIcon:
        """Loads an icon from the application's resource directory."""
        if status_key in self.STATUS_INFO:
            icon_path = self.STATUS_INFO[status_key]["icon_path"]
            if Path(icon_path).exists():
                return QIcon(icon_path)
            else:
                theme_map = {
                    self.STATUS_OK: "dialog-ok-apply", self.STATUS_ERROR: "dialog-error",
                    self.STATUS_WARNING: "dialog-warning", self.STATUS_PROCESSING: "view-refresh"
                }
                if status_key in theme_map:
                    return QIcon.fromTheme(theme_map[status_key])
        return QIcon()

    def add_file(self, filepath: Union[str, Path], status: str = STATUS_PENDING, status_message: Optional[str] = None):
        path_obj = Path(filepath)
        item = QListWidgetItem(path_obj.name)
        item.setData(Qt.ItemDataRole.UserRole, str(path_obj.resolve()))
        self.addItem(item)
        self.update_item_status(item, status, status_message)
        logger.debug(f"Added file to list: {path_obj.resolve()} with status: {status}")

    def update_item_status(self, item_or_filepath: Union[QListWidgetItem, str, Path], new_status: str, status_message: Optional[str] = None):
        item = self._find_item(item_or_filepath)
        if not item:
            logger.warning(f"Could not find item '{item_or_filepath}' to update status.")
            return

        # Store data required for re-translation
        item.setData(self.StatusRole, new_status)
        item.setData(self.StatusMessageRole, status_message)

        item.setIcon(self._get_status_icon(new_status))
        item.setToolTip(self._generate_tooltip_text(item))

        status_info = self.STATUS_INFO.get(new_status, {})
        item.setForeground(status_info.get("color", QColor("black")))
        
        full_path = item.data(Qt.ItemDataRole.UserRole)
        logger.debug(f"Updated status for {full_path} to {new_status}")

    def add_files(self, filepaths: List[Union[str, Path]], initial_status: str = STATUS_PENDING):
        for fp in filepaths:
            self.add_file(fp, status=initial_status)

    def get_all_filepaths(self) -> List[str]:
        return [self.item(i).data(Qt.ItemDataRole.UserRole) for i in range(self.count())]

    def remove_selected_files(self):
        selected = self.selectedItems()
        if not selected:
            return
        
        reply = QMessageBox.question(
            self, 
            self.tr("Confirm Removal"), 
            self.tr("Are you sure you want to remove {n} file(s)?").format(n=len(selected)),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, 
            QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            for item in selected:
                filepath = item.data(Qt.ItemDataRole.UserRole)
                self.takeItem(self.row(item))
                self.file_removed.emit(filepath)
                logger.info(f"Removed file from list: {filepath}")

    def clear_all_files(self):
        if self.count() == 0:
            return
        
        reply = QMessageBox.question(
            self, 
            self.tr("Confirm Clear"), 
            self.tr("Are you sure you want to remove all files from the list?"),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, 
            QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            super().clear()
            self.files_cleared.emit()
            logger.info("Cleared all files from the list.")

    def _show_context_menu(self, position):
        menu = QMenu(self)
        selected_items = self.selectedItems()

        if selected_items:
            remove_action = menu.addAction(QIcon.fromTheme("edit-delete"), self.tr("Remove Selected"))
            remove_action.triggered.connect(self.remove_selected_files)

            if len(selected_items) == 1:
                menu.addSeparator()
                item = selected_items[0]
                filepath = Path(item.data(Qt.ItemDataRole.UserRole))
                
                show_action = menu.addAction(QIcon.fromTheme("system-file-manager"), self.tr("Show in File Explorer"))
                show_action.triggered.connect(lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(filepath.parent))))
        
        menu.addSeparator()
        clear_all_action = menu.addAction(QIcon.fromTheme("edit-clear"), self.tr("Clear All Files"))
        clear_all_action.triggered.connect(self.clear_all_files)
        
        menu.exec(self.mapToGlobal(position))

    def _handle_item_double_click(self, item: QListWidgetItem):
        filepath = item.data(Qt.ItemDataRole.UserRole)
        self.file_double_clicked.emit(filepath)
        logger.debug(f"File item double-clicked: {filepath}")

    def _find_item(self, item_or_filepath: Union[QListWidgetItem, str, Path]) -> Optional[QListWidgetItem]:
        if isinstance(item_or_filepath, QListWidgetItem):
            return item_or_filepath
        
        path_to_find = str(Path(item_or_filepath).resolve())
        for i in range(self.count()):
            item = self.item(i)
            if item.data(Qt.ItemDataRole.UserRole) == path_to_find:
                return item
        return None