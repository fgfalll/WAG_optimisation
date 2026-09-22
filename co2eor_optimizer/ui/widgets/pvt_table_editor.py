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
from PyQt6.QtCore import pyqtSignal

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
        self.headers = default_headers or ["Pressure (psia)", "Rs (scf/STB)", "Bo (RB/STB)", "Oil Viscosity (cP)"]
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self.table_widget = QTableWidget()
        self.table_widget.setAlternatingRowColors(True)
        self.table_widget.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table_widget.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table_widget.itemChanged.connect(self._on_item_changed)
        self.set_headers(self.headers)
        self._layout.addWidget(self.table_widget)
        self._button_layout = QHBoxLayout()
        self.add_row_button = QPushButton(QIcon.fromTheme("list-add"), " Add Row")
        self.remove_row_button = QPushButton(QIcon.fromTheme("list-remove"), " Remove Selected")
        self.import_csv_button = QPushButton(QIcon.fromTheme("document-open"), " Import CSV")
        self.export_csv_button = QPushButton(QIcon.fromTheme("document-save"), " Export CSV")
        self._button_layout.addWidget(self.add_row_button); self._button_layout.addWidget(self.remove_row_button)
        self._button_layout.addStretch(1)
        self._button_layout.addWidget(self.import_csv_button); self._button_layout.addWidget(self.export_csv_button)
        self._layout.addLayout(self._button_layout)
        self.add_row_button.clicked.connect(self.add_row)
        self.remove_row_button.clicked.connect(self.remove_selected_rows)
        self.import_csv_button.clicked.connect(self.import_from_csv)
        self.export_csv_button.clicked.connect(self.export_to_csv)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumHeight(150)

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
                # After validating a row, we should also re-validate the next row
                # as its logic (e.g. pressure must be increasing) might now be affected.
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
        Returns True if the entire row is valid, False otherwise.
        """
        row_values: Dict[int, float] = {}
        has_invalid_number = False

        for col_index in range(self.table_widget.columnCount()):
            item = self.table_widget.item(row_index, col_index)
            if item is None: return False 

            val_str = item.text().strip().replace(',', '.')
            if not val_str:
                item.setBackground(QColor("#FFE0E0")) # Light Red
                item.setToolTip("Cell cannot be empty.")
                has_invalid_number = True
                continue
            
            try:
                row_values[col_index] = float(val_str)
                item.setBackground(QColor(0,0,0,0)) # Clear background
                item.setToolTip("")
            except ValueError:
                item.setBackground(QColor("#FFE0E0")) # Light Red
                item.setToolTip(f"Invalid input: '{item.text()}' is not a valid number.")
                has_invalid_number = True

        if has_invalid_number:
            self.validation_status.emit(f"Row {row_index + 1} has invalid or empty cells.", "error")
            return False

        # --- Logical validation ---
        is_row_logically_valid = True
        for col_index, current_value in row_values.items():
            item = self.table_widget.item(row_index, col_index)

            # 1. Range Check
            header_item = self.table_widget.horizontalHeaderItem(col_index)
            if header_item:
                header_text = header_item.text().lower()
                for key, (min_val, max_val) in self.VALIDATION_RANGES.items():
                    if key in header_text and not (min_val <= current_value <= max_val):
                        item.setBackground(QColor("#FFF3CD")) # Light Yellow
                        item.setToolTip(f"Warning: Value {current_value} is outside the typical range of {min_val} to {max_val}.")
                        is_row_logically_valid = False

            # 2. Monotonicity Check (for pressure, the first column)
            if col_index == 0 and row_index > 0:
                prev_item = self.table_widget.item(row_index - 1, 0)
                if prev_item and prev_item.text().strip():
                    try:
                        previous_value = float(prev_item.text().strip().replace(',', '.'))
                        if current_value <= previous_value:
                            item.setBackground(QColor("#FFF3CD")) # Light Yellow
                            item.setToolTip(f"Warning: Pressure {current_value} should be greater than the value in the row above ({previous_value}).")
                            is_row_logically_valid = False
                    except ValueError:
                        pass
        
        if not is_row_logically_valid:
            self.validation_status.emit(f"Row {row_index + 1} has out-of-range values (check tooltips).", "warning")
            return False
        
        self.validation_status.emit(f"Row {row_index + 1} is valid.", "ok")
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
            QMessageBox.information(self, "No Rows Selected", "Please select one or more rows to remove.")
            return
        reply = QMessageBox.question(self, "Confirm Removal", f"Remove {len(selected_rows)} selected row(s)?",
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
        self.headers = headers
        self.table_widget.setColumnCount(len(headers))
        self.table_widget.setHorizontalHeaderLabels(headers)

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
                    item = QTableWidgetItem(str(cell_value))
                    self.table_widget.setItem(r_idx, c_idx, item)
            # Validate all rows after setting data
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
                    except (ValueError, TypeError): pass # Leaves NaN
        return data

    def import_from_csv(self):
        filepath, _ = QFileDialog.getOpenFileName(self, f"Import {self.table_name} from CSV", "", "CSV Files (*.csv);;All Files (*)")
        if not filepath: return
        try:
            df = pd.read_csv(filepath, header='infer', skipinitialspace=True)
            if df.empty:
                QMessageBox.warning(self, "Import Warning", "The selected CSV file is empty."); return
            self.set_headers(list(df.columns))
            self.set_full_data(df.values)
            QMessageBox.information(self, "Import Successful", f"Loaded {len(df)} rows from\n{filepath}")
        except Exception as e:
            QMessageBox.critical(self, "Import Error", f"Could not import CSV file:\n{e}")

    def export_to_csv(self):
        if self.table_widget.rowCount() == 0:
            QMessageBox.information(self, "Export CSV", "Table is empty. Nothing to export."); return
        filepath, _ = QFileDialog.getSaveFileName(self, f"Export {self.table_name} to CSV", f"{self.table_name.replace(' ', '_')}.csv", "CSV Files (*.csv)")
        if not filepath: return
        try:
            data_np = self.get_data_as_numpy()
            current_headers = [self.table_widget.horizontalHeaderItem(i).text() for i in range(self.table_widget.columnCount())]
            df = pd.DataFrame(data_np, columns=current_headers)
            df.to_csv(filepath, index=False, na_rep='NaN')
            QMessageBox.information(self, "Export Successful", f"Table data exported to:\n{filepath}")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Could not export table data:\n{e}")