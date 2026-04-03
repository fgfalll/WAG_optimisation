import logging
from typing import List, Optional, Union, Dict, Tuple
import numpy as np
import pandas as pd

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTableWidget, QTableWidgetItem,
    QPushButton, QAbstractItemView, QHeaderView, QFileDialog, QMessageBox,
    QSizePolicy
)
from PyQt6.QtGui import QIcon, QColor
from PyQt6.QtCore import pyqtSignal, QEvent

logger = logging.getLogger(__name__)

class PVTTableEditorWidget(QWidget):
    """
    A widget for editing tabular data (e.g., PVT tables).
    Supports adding/removing rows, editing, and CSV import/export with live validation.
    """
    data_changed = pyqtSignal()
    validation_status = pyqtSignal(str, str) # message, level ('ok', 'warning', 'error')

    VALIDATION_RANGES: Dict[str, Tuple[float, float]] = {
        "pressure": (0, 20000), "rs": (0, 10000), "bo": (0.9, 3.0),
        "oil viscosity": (0.1, 100.0), "bg": (0.1, 5.0),
        "gas viscosity": (0.01, 0.1), "bw": (0.9, 1.2), "cw": (1e-7, 1e-5)
    }

    def __init__(self,
                 table_name: str = "PVT Table",
                 default_headers: Optional[List[str]] = None,
                 parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.table_name = table_name
        # Store untranslated headers
        self.untranslated_headers = default_headers or ["Pressure (psia)", "Rs (scf/STB)", "Bo (RB/STB)", "Oil Viscosity (cP)"]
        
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self.table_widget = QTableWidget()
        self.table_widget.setAlternatingRowColors(True)
        self.table_widget.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table_widget.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table_widget.itemChanged.connect(self._on_item_changed)
        
        self._layout.addWidget(self.table_widget)
        self._button_layout = QHBoxLayout()
        
        self.add_row_button = QPushButton()
        self.remove_row_button = QPushButton()
        self.import_csv_button = QPushButton()
        self.export_csv_button = QPushButton()
        
        self._button_layout.addWidget(self.add_row_button)
        self._button_layout.addWidget(self.remove_row_button)
        self._button_layout.addStretch(1)
        self._button_layout.addWidget(self.import_csv_button)
        self._button_layout.addWidget(self.export_csv_button)
        self._layout.addLayout(self._button_layout)
        
        self.add_row_button.clicked.connect(self.add_row)
        self.remove_row_button.clicked.connect(self.remove_selected_rows)
        self.import_csv_button.clicked.connect(self.import_from_csv)
        self.export_csv_button.clicked.connect(self.export_to_csv)
        
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumHeight(150)

        # Set initial text and translations
        self.retranslateUi()

    def retranslateUi(self):
        """
        Updates all translatable text in the widget.
        """
        # Set button text
        self.add_row_button.setText(self.tr(" Add Row"))
        self.remove_row_button.setText(self.tr(" Remove Selected"))
        self.import_csv_button.setText(self.tr(" Import CSV"))
        self.export_csv_button.setText(self.tr(" Export CSV"))

        # Re-translate and set headers
        translated_headers = [self.tr(h) for h in self.untranslated_headers]
        self.table_widget.setHorizontalHeaderLabels(translated_headers)
        
        # Re-validate all rows to update tooltips with new translations
        self._revalidate_all_rows()

    def changeEvent(self, event: QEvent):
        """
        Handles language change events to re-translate the UI.
        """
        if event.type() == QEvent.Type.LanguageChange:
            self.retranslateUi()
        super().changeEvent(event)

    def _revalidate_all_rows(self):
        """
        Force re-validation of all rows, which updates cell tooltips.
        """
        self.table_widget.blockSignals(True)
        try:
            for row_idx in range(self.table_widget.rowCount()):
                self._validate_row(row_idx)
        finally:
            self.table_widget.blockSignals(False)

    def _on_item_changed(self, item: QTableWidgetItem):
        """
        Trigger for validating the entire row an item belongs to.
        """
        if item is None:
            return
        
        self.table_widget.blockSignals(True)
        try:
            changed_row = item.row()
            is_row_valid = self._validate_row(changed_row)

            if is_row_valid:
                if changed_row + 1 < self.table_widget.rowCount():
                    self._validate_row(changed_row + 1)
                self.data_changed.emit()

        except Exception as e:
            logger.error(f"Unexpected error in _on_item_changed for table '{self.table_name}'.", exc_info=True)
        finally:
            self.table_widget.blockSignals(False)

    def _validate_row(self, row_index: int) -> bool:
        """
        Validates all cells in a given row. Sets background colors and tooltips.
        """
        for col_index in range(self.table_widget.columnCount()):
            item = self.table_widget.item(row_index, col_index)
            if item is None: return False 

            val_str = item.text().strip().replace(',', '.')
            item.setBackground(QColor(0,0,0,0)) 
            item.setToolTip("")

            if not val_str:
                item.setBackground(QColor("#FFE0E0"))
                item.setToolTip(self.tr("Error: Cell cannot be empty."))
                self.validation_status.emit(self.tr("Row {0} has an empty cell.").format(row_index + 1), "error")
                return False

            try:
                float(val_str)
            except ValueError:
                item.setBackground(QColor("#FFE0E0"))
                item.setToolTip(self.tr("Error: '{0}' is not a valid number.").format(item.text()))
                self.validation_status.emit(self.tr("Row {0} has a non-numeric value.").format(row_index + 1), "error")
                return False

        has_warning = False
        for col_index in range(self.table_widget.columnCount()):
            item = self.table_widget.item(row_index, col_index)
            current_value = float(item.text().strip().replace(',', '.'))
            
            header_item = self.table_widget.horizontalHeaderItem(col_index)
            if not header_item: continue
            # Use untranslated headers for logic checks
            header_text = self.untranslated_headers[col_index].lower()

            for key, (min_val, max_val) in self.VALIDATION_RANGES.items():
                if key in header_text and not (min_val <= current_value <= max_val):
                    item.setBackground(QColor("#FFF3CD"))
                    item.setToolTip(self.tr("Warning: Value {0} is outside the typical range of {1} to {2}.").format(current_value, min_val, max_val))
                    has_warning = True

            if 'pressure' in header_text and row_index > 0:
                prev_item = self.table_widget.item(row_index - 1, col_index)
                if prev_item and prev_item.text().strip():
                    try:
                        previous_value = float(prev_item.text().strip().replace(',', '.'))
                        if current_value <= previous_value:
                            item.setBackground(QColor("#FFF3CD"))
                            item.setToolTip(self.tr("Warning: Pressure {0} should be greater than the value in the row above ({1}).").format(current_value, previous_value))
                            has_warning = True
                    except ValueError:
                        pass
        
        if has_warning:
            self.validation_status.emit(self.tr("Row {0} has warnings (check tooltips on yellow cells).").format(row_index + 1), "warning")
        else:
            self.validation_status.emit(self.tr("Row {0} is valid.").format(row_index + 1), "ok")
        
        return True

    def add_row(self):
        self.table_widget.blockSignals(True)
        try:
            row_position = self.table_widget.rowCount()
            self.table_widget.insertRow(row_position)
            for col in range(self.table_widget.columnCount()):
                self.table_widget.setItem(row_position, col, QTableWidgetItem(""))
        finally:
            self.table_widget.blockSignals(False)
        self.data_changed.emit()

    def remove_selected_rows(self):
        selected_rows = sorted(list(set(idx.row() for idx in self.table_widget.selectedIndexes())), reverse=True)
        if not selected_rows:
            QMessageBox.information(self, self.tr("No Rows Selected"), self.tr("Please select one or more rows to remove."))
            return
        reply = QMessageBox.question(self, self.tr("Confirm Removal"), self.tr("Remove {0} selected row(s)?").format(len(selected_rows)),
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            self.table_widget.blockSignals(True)
            try:
                for row_idx in selected_rows:
                    self.table_widget.removeRow(row_idx)
            finally:
                self.table_widget.blockSignals(False)
            self.data_changed.emit()

    def set_headers(self, headers: List[str]):
        self.untranslated_headers = headers
        self.table_widget.setColumnCount(len(headers))
        translated_headers = [self.tr(h) for h in headers]
        self.table_widget.setHorizontalHeaderLabels(translated_headers)

    def set_full_data(self, data: Union[List[list], np.ndarray]):
        self.table_widget.blockSignals(True)
        try:
            self.table_widget.setRowCount(0)
            data_array = np.asarray(data)
            if data_array.ndim != 2 or data_array.size == 0:
                self.data_changed.emit(); return
            num_rows, num_cols = data_array.shape
            self.table_widget.setRowCount(num_rows)
            if num_cols != self.table_widget.columnCount():
                new_headers = [f"Column {j+1}" for j in range(num_cols)]
                self.set_headers(new_headers)
            for r_idx, row_list in enumerate(data_array):
                for c_idx, cell_value in enumerate(row_list):
                    item = QTableWidgetItem(str(cell_value) if not pd.isna(cell_value) else "")
                    self.table_widget.setItem(r_idx, c_idx, item)
            
            for r_idx in range(num_rows):
                self._validate_row(r_idx)
        finally:
            self.table_widget.blockSignals(False)
        self.data_changed.emit()

    def get_data_as_numpy(self, dtype=float) -> np.ndarray:
        rows, cols = self.table_widget.rowCount(), self.table_widget.columnCount()
        if rows == 0 or cols == 0:
            return np.empty((0, cols), dtype=dtype)
        data = np.full((rows, cols), np.nan, dtype=dtype)
        for r in range(rows):
            for c in range(cols):
                item = self.table_widget.item(r, c)
                if item and item.text().strip():
                    try:
                        data[r, c] = dtype(item.text().strip().replace(',', '.'))
                    except (ValueError, TypeError) as e:
                        # Import the centralized error manager
                        import sys
                        import os
                        sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
                        from error_handler import handle_caught_exception, ErrorSeverity, ErrorCategory

                        # Handle the conversion error properly instead of silently ignoring it
                        handle_caught_exception(
                            operation=f"convert table cell value at position ({r}, {c})",
                            exception=e,
                            context={
                                "row": r,
                                "column": c,
                                "input_value": item.text().strip(),
                                "target_type": str(dtype),
                                "table_name": self.table_name
                            },
                            user_action_suggested=f"Check value format at cell ({r+1}, {c+1}). Expected {dtype.__name__} type.",
                            show_dialog=False,  # Don't interrupt user for individual cell errors
                            severity=ErrorSeverity.WARNING,
                            category=ErrorCategory.DATA
                        )
                        # Set to None to indicate invalid data instead of silently ignoring
                        data[r, c] = None
        return data

    def import_from_csv(self):
        title = self.tr("Import {0} from CSV").format(self.table_name)
        filter = self.tr("CSV Files (*.csv);;All Files (*)")
        filepath, _ = QFileDialog.getOpenFileName(self, title, "", filter)
        if not filepath: return
        try:
            df = pd.read_csv(filepath, header='infer', skipinitialspace=True)
            if df.empty:
                QMessageBox.warning(self, self.tr("Import Warning"), self.tr("The selected CSV file is empty.")); return
            self.set_headers(list(df.columns))
            self.set_full_data(df.values)
            QMessageBox.information(self, self.tr("Import Successful"), self.tr("Loaded {0} rows from\n{1}").format(len(df), filepath))
        except Exception as e:
            QMessageBox.critical(self, self.tr("Import Error"), self.tr("Could not import CSV file:\n{0}").format(e))

    def export_to_csv(self):
        if self.table_widget.rowCount() == 0:
            QMessageBox.information(self, self.tr("Export CSV"), self.tr("Table is empty. Nothing to export.")); return
        
        title = self.tr("Export {0} to CSV").format(self.table_name)
        filename = f"{self.table_name.replace(' ', '_')}.csv"
        filter = self.tr("CSV Files (*.csv)")
        filepath, _ = QFileDialog.getSaveFileName(self, title, filename, filter)
        if not filepath: return
        
        try:
            data_np = self.get_data_as_numpy()
            current_headers = [self.table_widget.horizontalHeaderItem(i).text() for i in range(self.table_widget.columnCount())]
            df = pd.DataFrame(data_np, columns=current_headers)
            df.to_csv(filepath, index=False, na_rep='NaN')
            QMessageBox.information(self, self.tr("Export Successful"), self.tr("Table data exported to:\n{0}").format(filepath))
        except Exception as e:
            QMessageBox.critical(self, self.tr("Export Error"), self.tr("Could not export table data:\n{0}").format(e))