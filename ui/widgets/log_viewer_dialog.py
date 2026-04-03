import logging
from typing import List, Optional, Dict

import numpy as np
from PyQt6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QComboBox,
    QListWidget,
    QListWidgetItem,
    QDialogButtonBox,
    QSplitter,
    QGroupBox,
    QWidget,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QFrame,
    QFormLayout,
    QTabWidget,
    QTableWidget,
)
from PyQt6.QtGui import QIcon
from PyQt6.QtCore import Qt, QEvent

# This requires matplotlib to be installed (pip install matplotlib)
try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except ImportError:

    class FigureCanvas(QWidget):
        pass

    class Figure(object):
        pass

    plt = None
    MATPLOTLIB_AVAILABLE = False
    logging.warning("Matplotlib not found. Log plotting will be disabled.")

try:
    from core.data_models import WellData
except ImportError:

    class WellData:
        pass

    logging.critical("LogViewerDialog: Could not import WellData model.")

logger = logging.getLogger(__name__)


class LogTrackWidget(QFrame):
    """A self-contained widget for a single log track with its controls."""

    def __init__(self, log_name: str, unit: str, parent=None):
        super().__init__(parent)
        self.log_name = log_name
        self.unit = unit
        self.setFrameShape(QFrame.Shape.StyledPanel)

        self._setup_ui()
        self.retranslateUi()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)

        # Title and controls
        header_layout = QHBoxLayout()
        self.title_label = QLabel()
        self.update_title()  # Set initial title

        header_layout.addWidget(self.title_label)
        header_layout.addStretch()
        self.scale_label = QLabel()
        header_layout.addWidget(self.scale_label)
        self.min_scale_edit = QLineEdit()
        self.min_scale_edit.setFixedWidth(50)
        self.max_scale_edit = QLineEdit()
        self.max_scale_edit.setFixedWidth(50)
        header_layout.addWidget(self.min_scale_edit)
        header_layout.addWidget(self.max_scale_edit)
        layout.addLayout(header_layout)

        # Matplotlib canvas for this track
        self.figure = Figure(figsize=(2, 10))  # Tall and narrow for a track
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        layout.addWidget(self.canvas)

    def update_title(self):
        """Updates the title label of the track."""
        self.title_label.setText(f"<b>{self.log_name}</b> ({self.unit})")

    def retranslateUi(self):
        """Retranslates the static UI elements of the widget."""
        self.scale_label.setText(self.tr("Scale:"))
        self.update_title()  # Also update the title in case the log_name was a translatable string like "Depth"

    def changeEvent(self, event: QEvent):
        """Handle language change events to re-translate the UI."""
        if event.type() == QEvent.Type.LanguageChange:
            self.retranslateUi()
        super().changeEvent(event)


class WellViewerDialog(QDialog):
    """
    An interactive, multi-track dialog for visualizing and annotating well logs.
    """

    def __init__(
        self, well_data_list: List[WellData], selected_well: Optional[WellData] = None, parent=None
    ):
        super().__init__(parent)
        self.well_data_list = well_data_list
        self.current_well: Optional[WellData] = selected_well
        self.annotations: List[plt.Annotation] = []
        self.is_adding_note = False
        self.cid = None
        self.depth_track_widget: Optional[LogTrackWidget] = None

        self.setMinimumSize(1000, 800)

        self._setup_ui()
        self._connect_signals()
        self.retranslateUi()

        if self.well_data_list:
            self._populate_well_combobox()
            if self.current_well:
                index = self.well_combo.findData(self.current_well)
                if index != -1:
                    self.well_combo.setCurrentIndex(index)

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)

        top_bar_layout = QHBoxLayout()
        self.select_well_label = QLabel()
        top_bar_layout.addWidget(self.select_well_label)
        self.well_combo = QComboBox()
        top_bar_layout.addWidget(self.well_combo, 1)
        top_bar_layout.addStretch()
        self.add_note_btn = QPushButton(QIcon.fromTheme("edit-add"), "")
        self.add_note_btn.setCheckable(True)
        top_bar_layout.addWidget(self.add_note_btn)
        main_layout.addLayout(top_bar_layout)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left side: Log selection list
        self.logs_group = QGroupBox()
        logs_layout = QVBoxLayout(self.logs_group)
        self.log_list = QListWidget()
        self.log_list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        logs_layout.addWidget(self.log_list)

        self.metadata_group = QGroupBox()
        self.metadata_layout = QFormLayout(self.metadata_group)
        logs_layout.addWidget(self.metadata_group)

        splitter.addWidget(self.logs_group)

        # Right side: Tabbed view for logs and other data
        self.right_tabs = QTabWidget()
        splitter.addWidget(self.right_tabs)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.tracks_container = QWidget()
        self.tracks_layout = QHBoxLayout(self.tracks_container)
        self.tracks_layout.setSpacing(0)
        self.scroll_area.setWidget(self.tracks_container)
        self.right_tabs.addTab(self.scroll_area, self.tr("Log Tracks"))

        self.perforations_group = QGroupBox(self.tr("Perforations"))
        self.perforations_layout = QVBoxLayout(self.perforations_group)
        self.perforations_table = QTableWidget()
        self.perforations_table.setColumnCount(2)
        self.perforations_table.setHorizontalHeaderLabels([self.tr("Top MD"), self.tr("Bottom MD")])
        self.perforations_layout.addWidget(self.perforations_table)
        self.right_tabs.addTab(self.perforations_group, self.tr("Perforations"))

        splitter.addWidget(self.logs_group)

        splitter.setSizes([200, 800])
        main_layout.addWidget(splitter, 1)

        self.button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        main_layout.addWidget(self.button_box)

    def retranslateUi(self):
        """Retranslates all UI elements in the dialog."""
        self.setWindowTitle(self.tr("Well Viewer"))
        self.select_well_label.setText(self.tr("Select Well:"))
        self.add_note_btn.setText(self.tr("Add Note"))
        self.logs_group.setTitle(self.tr("Available Logs"))
        self.metadata_group.setTitle(self.tr("Well Information"))

        self.right_tabs.setTabText(0, self.tr("Log Tracks"))
        self.right_tabs.setTabText(1, self.tr("Perforations"))

        # If a depth track exists, update its name with the translated version
        if self.depth_track_widget:
            self.depth_track_widget.log_name = self.tr("Depth")
            self.depth_track_widget.update_title()

    def changeEvent(self, event: QEvent):
        """Handle language change events to re-translate the UI."""
        if event.type() == QEvent.Type.LanguageChange:
            self.retranslateUi()
        super().changeEvent(event)

    def _connect_signals(self):
        self.well_combo.currentIndexChanged.connect(self._on_well_changed)
        self.log_list.itemSelectionChanged.connect(self._update_tracks_display)
        self.add_note_btn.toggled.connect(self._toggle_add_note_mode)
        self.button_box.rejected.connect(self.reject)

    def _populate_well_combobox(self):
        self.well_combo.blockSignals(True)
        self.well_combo.clear()
        for well_data in self.well_data_list:
            self.well_combo.addItem(well_data.name, userData=well_data)
        self.well_combo.blockSignals(False)
        if self.well_combo.count() > 0:
            self.well_combo.setCurrentIndex(0)
            self._on_well_changed(0)

    def _on_well_changed(self, index: int):
        if index < 0:
            return
        self.current_well = self.well_combo.itemData(index)
        self.annotations.clear()

        # Clear previous metadata
        while self.metadata_layout.count():
            self.metadata_layout.takeAt(0).widget().deleteLater()

        if self.current_well:
            self._populate_log_list(self.current_well)

            # Populate metadata
            self.metadata_layout.addRow(self.tr("Name:"), QLabel(self.current_well.name))
            self.metadata_layout.addRow(
                self.tr("Status:"), QLabel(self.current_well.metadata.get("status", "N/A"))
            )
            self.metadata_layout.addRow(
                self.tr("Depth Range:"),
                QLabel(f"{self.current_well.depths[0]:.1f} - {self.current_well.depths[-1]:.1f}"),
            )
            self.metadata_layout.addRow(
                self.tr("API Gravity:"), QLabel(str(self.current_well.metadata.get("API", "N/A")))
            )
            self.metadata_layout.addRow(
                self.tr("Temperature:"),
                QLabel(str(self.current_well.metadata.get("Temperature", "N/A"))),
            )

            # Populate perforations
            self.perforations_table.setRowCount(0)
            if self.current_well.perforation_properties:
                for perf in self.current_well.perforation_properties:
                    row = self.perforations_table.rowCount()
                    self.perforations_table.insertRow(row)
                    self.perforations_table.setItem(
                        row, 0, QTableWidgetItem(str(perf.get("top", "N/A")))
                    )
                    self.perforations_table.setItem(
                        row, 1, QTableWidgetItem(str(perf.get("bottom", "N/A")))
                    )

        self._update_tracks_display()

    def _populate_log_list(self, well_data: WellData):
        self.log_list.blockSignals(True)
        self.log_list.clear()
        sorted_logs = sorted([log for log in well_data.properties.keys() if log.upper() != "DEPT"])
        for log_name in sorted_logs:
            unit = well_data.units.get(log_name, "N/A")
            item = QListWidgetItem(f"{log_name} ({unit})")
            item.setData(Qt.ItemDataRole.UserRole, log_name)
            self.log_list.addItem(item)
        self.log_list.blockSignals(False)

    def _update_tracks_display(self):
        if not MATPLOTLIB_AVAILABLE or not self.current_well:
            return

        # Clear existing tracks
        self.depth_track_widget = None
        while self.tracks_layout.count():
            child = self.tracks_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        selected_items = self.log_list.selectedItems()
        self.add_note_btn.setEnabled(bool(selected_items))
        if not selected_items:
            return

        y_data = self.current_well.depths

        # Add Depth Track
        self.depth_track_widget = self._create_depth_track(y_data)
        self.tracks_layout.addWidget(self.depth_track_widget)

        # Add perforation highlights to depth track
        if self.current_well.perforation_properties:
            for perf in self.current_well.perforation_properties:
                top = perf.get("top")
                bottom = perf.get("bottom")
                if top is not None and bottom is not None:
                    self.depth_track_widget.ax.axhspan(top, bottom, color="green", alpha=0.2)

        # Add selected log tracks
        for item in selected_items:
            log_name = item.data(Qt.ItemDataRole.UserRole)
            unit = self.current_well.units.get(log_name, "")
            x_data = self.current_well.properties.get(log_name)

            track_widget = LogTrackWidget(log_name, unit)
            ax = track_widget.ax

            ax.plot(x_data, y_data)
            ax.set_ylim(y_data.max(), y_data.min())  # Invert Y-axis
            ax.grid(True, linestyle=":", alpha=0.7)
            ax.tick_params(axis="y", which="both", left=False, labelleft=False)  # Hide y-ticks

            # Set initial scale
            x_min, x_max = np.nanmin(x_data), np.nanmax(x_data)
            track_widget.min_scale_edit.setText(f"{x_min:.1f}")
            track_widget.max_scale_edit.setText(f"{x_max:.1f}")
            ax.set_xlim(x_min, x_max)

            track_widget.min_scale_edit.editingFinished.connect(
                lambda w=track_widget: self._update_track_scale(w)
            )
            track_widget.max_scale_edit.editingFinished.connect(
                lambda w=track_widget: self._update_track_scale(w)
            )

            self.tracks_layout.addWidget(track_widget)
            track_widget.canvas.draw()

    def _create_depth_track(self, y_data: np.ndarray) -> LogTrackWidget:
        track_widget = LogTrackWidget(self.tr("Depth"), "ft")
        ax = track_widget.ax
        ax.set_ylim(y_data.max(), y_data.min())
        ax.grid(True, linestyle=":", alpha=0.7)
        ax.set_xticks([])  # No x-ticks for depth track
        track_widget.scale_label.setVisible(False)
        track_widget.min_scale_edit.setVisible(False)
        track_widget.max_scale_edit.setVisible(False)
        track_widget.figure.tight_layout()
        track_widget.canvas.draw()
        return track_widget

    def _update_track_scale(self, track_widget: LogTrackWidget):
        try:
            xmin = float(track_widget.min_scale_edit.text())
            xmax = float(track_widget.max_scale_edit.text())
            if xmin < xmax:
                track_widget.ax.set_xlim(xmin, xmax)
                track_widget.canvas.draw()
        except ValueError:
            pass  # Ignore non-numeric input

    def _toggle_add_note_mode(self, checked: bool):
        if not self.tracks_layout.count():
            self.add_note_btn.setChecked(False)
            return

        first_track = (
            self.tracks_layout.itemAt(1).widget() if self.tracks_layout.count() > 1 else None
        )
        if not first_track:
            return

        self.is_adding_note = checked
        if checked:
            self.setCursor(Qt.CursorShape.CrossCursor)
            self.cid = first_track.canvas.mpl_connect("button_press_event", self._on_plot_click)
        else:
            self.setCursor(Qt.CursorShape.ArrowCursor)
            if self.cid:
                first_track.canvas.mpl_disconnect(self.cid)
                self.cid = None

    def _on_plot_click(self, event):
        if not self.is_adding_note or event.inaxes is None:
            return

        depth = event.ydata
        note_text = self.tr("Note @ {0:.1f} ft").format(depth)

        # Add annotation to all visible tracks
        for i in range(1, self.tracks_layout.count()):
            track_widget = self.tracks_layout.itemAt(i).widget()
            ax = track_widget.ax

            # Place annotation in the middle of the x-axis
            x_pos = (ax.get_xlim()[0] + ax.get_xlim()[1]) / 2

            ann = ax.annotate(
                note_text,
                xy=(x_pos, depth),
                xytext=(10, 10),
                textcoords="offset points",
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"),
                bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.7),
            )
            self.annotations.append(ann)
            track_widget.canvas.draw()

        # Deactivate note mode after adding one
        self.add_note_btn.setChecked(False)
