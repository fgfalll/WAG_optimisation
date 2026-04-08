import logging
from typing import Optional, Dict, Any, Type, List
import numpy as np
from PyQt6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QGroupBox,
    QLabel,
    QPushButton,
    QMessageBox,
    QLineEdit,
    QDialogButtonBox,
    QWidget,
    QFormLayout,
    QTableWidget,
    QTableWidgetItem,
    QAbstractItemView,
    QComboBox,
)
from PyQt6.QtGui import QIcon
from PyQt6.QtCore import QPointF, pyqtSignal, QEvent

try:
    from .parameter_input_group import ParameterInputGroup
except ImportError:

    class ParameterInputGroup(QWidget):
        finalValueChanged = pyqtSignal(object)
        param_name: str = ""

        def __init__(self, param_name="", label_text="", input_type="", **kwargs):
            super().__init__()

        def get_value(self):
            return ""

        def set_value(self, v):
            pass

        def setEnabled(self, b):
            pass

        def setProperty(self, n, v):
            pass

        def property(self, n):
            pass

    logging.critical("ManualWellDialog: Failed to import ParameterInputGroup.")
try:
    from .depth_profile_dialog import DepthProfileDialog
except ImportError:

    class DepthProfileDialog(QDialog):
        pass

    logging.critical("ManualWellDialog: Failed to import DepthProfileDialog.")
try:
    from core.data_models import WellData
except ImportError:

    class WellData:
        pass

    logging.critical("ManualWellDialog: Could not import WellData model.")

logger = logging.getLogger(__name__)


class ManualWellDialog(QDialog):
    """A dialog to manually input a well with detailed perforations and an optional path editor."""

    KEY_PARAM_DEFS = {
        "SurfaceX": ("Surface X Coordinate (ft)", "lineedit", float, {"default_value": 0.0}),
        "SurfaceY": ("Surface Y Coordinate (ft)", "lineedit", float, {"default_value": 0.0}),
        "WellboreRadius": ("Wellbore Radius (ft)", "lineedit", float, {"default_value": 0.35}),
        "SkinFactor": ("Skin Factor", "lineedit", float, {"default_value": 0.0}),
        "TopDepth": ("Well Top Depth / MD (ft)", "lineedit", float, {"default_value": 1.0}),
        "BottomDepth": (
            "Well Bottom Depth / MD (ft)",
            "lineedit",
            float,
            {"default_value": 3000.0},
        ),
    }

    def __init__(self, existing_names: List[str] = [], parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setMinimumSize(600, 500)

        self.key_param_widgets: Dict[str, ParameterInputGroup] = {}
        self.key_param_values: Dict[str, Any] = {}
        self.well_path: List[QPointF] = []
        self.existing_names = existing_names

        main_layout = QVBoxLayout(self)

        name_layout = QHBoxLayout()
        self.well_name_label = QLabel()
        name_layout.addWidget(self.well_name_label)
        self.well_name_edit = QLineEdit()
        name_layout.addWidget(self.well_name_edit)
        main_layout.addLayout(name_layout)

        self.params_group = QGroupBox()
        params_form_layout = QFormLayout(self.params_group)
        for name, (label, w_type, p_type, kwargs) in self.KEY_PARAM_DEFS.items():
            # NOTE: The label for this custom widget is set at creation. For dynamic
            # translation, the ParameterInputGroup class would need to be modified.
            widget = ParameterInputGroup(
                param_name=name, label_text=label, input_type=w_type, **kwargs
            )
            widget.setProperty("param_type", p_type)
            widget.finalValueChanged.connect(self._on_parameter_changed)
            self.key_param_widgets[name] = widget
            params_form_layout.addRow(widget)

        self.status_label = QLabel(self.tr("Status:"))
        self.status_combo = QComboBox()
        self.status_combo.addItems(["Active", "Inactive", "Injector"])
        params_form_layout.addRow(self.status_label, self.status_combo)

        self.edit_path_btn = QPushButton()
        self.edit_path_btn.setIcon(QIcon.fromTheme("document-edit"))
        params_form_layout.addRow(self.edit_path_btn)
        main_layout.addWidget(self.params_group)

        self.perf_group = QGroupBox()
        perf_layout = QVBoxLayout(self.perf_group)
        self.no_perf_warning_label = QLabel()
        self.no_perf_warning_label.setStyleSheet("color: #D32F2F;")

        perf_button_layout = QHBoxLayout()
        self.add_perf_btn = QPushButton(QIcon.fromTheme("list-add"), "Add")
        self.remove_perf_btn = QPushButton(QIcon.fromTheme("list-remove"), "Remove")
        perf_button_layout.addStretch()
        perf_button_layout.addWidget(self.add_perf_btn)
        perf_button_layout.addWidget(self.remove_perf_btn)

        self.perf_table = QTableWidget()
        self.perf_table.setColumnCount(2)
        self.perf_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        perf_layout.addWidget(self.no_perf_warning_label)
        perf_layout.addWidget(self.perf_table)
        perf_layout.addLayout(perf_button_layout)
        main_layout.addWidget(self.perf_group, 1)

        self.button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        main_layout.addWidget(self.button_box)

        self.edit_path_btn.clicked.connect(self._open_depth_profile_editor)
        self.add_perf_btn.clicked.connect(self._add_perforation_row)
        self.remove_perf_btn.clicked.connect(self._remove_perforation_row)
        self.perf_table.model().rowsInserted.connect(self._update_ui_state)
        self.perf_table.model().rowsRemoved.connect(self._update_ui_state)
        self.status_combo.currentIndexChanged.connect(self._update_well_name)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)

        self._initialize_defaults()
        self.retranslateUi()
        self._update_well_name()

    def _update_well_name(self):
        self.well_name_edit.setText(self._generate_default_name())

    def _generate_default_name(self) -> str:
        status = self.status_combo.currentText().lower()
        if status in ["active", "inactive"]:
            well_type = "producer"
        else:
            well_type = "injector"

        count = 1
        while f"well-{well_type}-{count}" in self.existing_names:
            count += 1
        return f"Well-{well_type.capitalize()}-{count}"

    def retranslateUi(self):
        """Updates all user-visible strings in the dialog to the current language."""
        self.setWindowTitle(self.tr("Add Well Manually"))
        self.well_name_label.setText(self.tr("Well Name:"))
        self.params_group.setTitle(self.tr("Key Well Parameters"))
        self.edit_path_btn.setText(self.tr(" Edit Depth Profile... (Optional)"))
        self.perf_group.setTitle(self.tr("Perforations"))
        self.no_perf_warning_label.setText(
            self.tr(
                "<b>Warning:</b> No perforations defined. The entire well path will be treated as connected to the reservoir."
            )
        )
        self.add_perf_btn.setText(self.tr("Add Perforation"))
        self.remove_perf_btn.setText(self.tr("Remove Selected"))
        self.perf_table.setHorizontalHeaderLabels(
            [self.tr("Top MD (ft)"), self.tr("Bottom MD (ft)")]
        )

    def changeEvent(self, event: QEvent):
        """Handle language change events to re-translate the UI."""
        if event.type() == QEvent.Type.LanguageChange:
            self.retranslateUi()
        super().changeEvent(event)

    def _initialize_defaults(self):
        """Initializes widgets with default values and populates the internal data dictionary."""
        for name, widget in self.key_param_widgets.items():
            param_def = self.KEY_PARAM_DEFS[name]
            default_value = param_def[3].get("default_value")
            p_type = param_def[2]

            widget.set_value(default_value)

            try:
                self.key_param_values[name] = self._coerce_value(str(default_value), p_type)
            except (ValueError, TypeError) as e:
                logger.error(f"Could not coerce default value for {name}: {e}")

        self._update_ui_state()

    def _coerce_value(self, value: Any, type_hint: Type) -> Any:
        try:
            if isinstance(value, str):
                value = value.replace(",", ".")
            if value is None or (isinstance(value, str) and not value.strip()):
                return None
            if type_hint is float:
                return float(value)
            if type_hint is int:
                return int(float(value))
            return value
        except (ValueError, TypeError):
            raise ValueError("Must be a valid number.")

    def _on_parameter_changed(self, value: Any):
        sender = self.sender()
        if not isinstance(sender, ParameterInputGroup):
            return
        param_name = sender.param_name
        try:
            p_type = self.KEY_PARAM_DEFS[param_name][2]
            self.key_param_values[param_name] = self._coerce_value(value, p_type)
            if param_name in ("TopDepth", "BottomDepth") and self.well_path:
                top = self.key_param_values.get("TopDepth", 0)
                bottom = self.key_param_values.get("BottomDepth", 0)
                if top is not None and bottom is not None and bottom > top:
                    self.well_path[0].setY(top)
                    self.well_path[-1].setY(bottom)
        except (ValueError, TypeError):
            if param_name in self.key_param_values:
                del self.key_param_values[param_name]

    def _open_depth_profile_editor(self):
        if not self.well_path:
            top = self.key_param_values.get("TopDepth")
            bottom = self.key_param_values.get("BottomDepth")
            if top is None or bottom is None or bottom <= top:
                QMessageBox.warning(
                    self,
                    self.tr("Invalid Depths"),
                    self.tr("Please set valid Top and Bottom depths before editing the profile."),
                )
                return
            self.well_path = [QPointF(0, top), QPointF(0, bottom)]

        dialog = DepthProfileDialog(self.well_path, self)
        if dialog.exec():
            self.well_path = dialog.get_path()
            logger.info("Well path updated from the editor dialog.")

    def _add_perforation_row(self):
        row_pos = self.perf_table.rowCount()
        self.perf_table.insertRow(row_pos)

    def _remove_perforation_row(self):
        selected_rows = sorted(
            {index.row() for index in self.perf_table.selectedIndexes()}, reverse=True
        )
        for row in selected_rows:
            self.perf_table.removeRow(row)

    def _update_ui_state(self):
        has_perforations = self.perf_table.rowCount() > 0
        if "SurfaceX" in self.key_param_widgets:
            self.key_param_widgets["SurfaceX"].setEnabled(not has_perforations)
        if "SurfaceY" in self.key_param_widgets:
            self.key_param_widgets["SurfaceY"].setEnabled(not has_perforations)
        self.no_perf_warning_label.setVisible(not has_perforations)

    def get_well_data(self) -> Optional[WellData]:
        well_name = self.well_name_edit.text().strip()
        if not well_name:
            QMessageBox.warning(self, self.tr("Input Error"), self.tr("A well name is required."))
            return None

        try:
            path_to_use = self.well_path

            if not path_to_use:
                top = self.key_param_values.get("TopDepth")
                bottom = self.key_param_values.get("BottomDepth")
                sx = self.key_param_values.get("SurfaceX", 0.0)
                sy = self.key_param_values.get("SurfaceY", 0.0)
                if top is None or bottom is None or bottom <= top:
                    raise ValueError(
                        self.tr(
                            "Could not create default well path. Please ensure Top and Bottom depths are valid and Top < Bottom."
                        )
                    )
                logger.info(
                    "No explicit path set. Creating default straight vertical well path with 10ft sampling."
                )
                depths_np = np.arange(top, bottom, 10.0)
                if depths_np.size == 0 or depths_np[-1] < bottom:
                    depths_np = np.append(depths_np, bottom)
                if depths_np.size == 0:
                    depths_np = np.array([top, bottom])
                # Create 3D path: [x, y, z] using SurfaceX and SurfaceY
                well_path_np = np.column_stack((np.full_like(depths_np, sx), np.full_like(depths_np, sy), depths_np))
            else:
                # Convert 2D path points [x, depth] to 3D [x, 0, depth]
                well_path_np = np.array([[p.x(), 0.0, p.y()] for p in path_to_use])

            # depths_np should be the unique z-coordinates
            depths_np = np.sort(np.unique(well_path_np[:, 2]))

            perfs = []
            for row in range(self.perf_table.rowCount()):
                try:
                    top = float(self.perf_table.item(row, 0).text())
                    bot = float(self.perf_table.item(row, 1).text())
                    if top >= bot:
                        raise ValueError(
                            self.tr("Row {0}: Top depth must be less than bottom depth.").format(
                                row + 1
                            )
                        )
                    perfs.append({"top": top, "bottom": bot})
                except (AttributeError, ValueError) as e:
                    raise ValueError(
                        self.tr("Invalid data in perforations table at row {0}: {1}").format(
                            row + 1, e
                        )
                    )

            final_metadata = self.key_param_values.copy()
            status = self.status_combo.currentText()
            final_metadata["status"] = status
            # Explicitly set well type for integration engine
            final_metadata["type"] = "injector" if status.lower() == "injector" else "producer"
            
            if not perfs:
                final_metadata["perforations_treatment"] = "entire_wellbore"

            well_props = {
                "WellboreRadius": [self.key_param_values.get("WellboreRadius", 0.35)],
                "SkinFactor": [self.key_param_values.get("SkinFactor", 0.0)],
            }

            return WellData(
                name=well_name,
                depths=depths_np,
                well_path=well_path_np,
                perforation_properties=perfs,
                metadata=final_metadata,
                properties=well_props,
                units={},
            )
        except Exception as e:
            QMessageBox.critical(
                self,
                self.tr("Data Creation Error"),
                self.tr("Could not create well data: {0}").format(e),
            )
            return None
