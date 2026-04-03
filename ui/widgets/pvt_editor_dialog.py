import logging
from typing import Optional, Dict, Any, List

import numpy as np
import pandas as pd
from PyQt6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QGroupBox,
    QLabel,
    QPushButton,
    QTabWidget,
    QMessageBox,
    QLineEdit,
    QComboBox,
    QDialogButtonBox,
    QWidget,
    QSplitter,
    QFormLayout,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QCompleter,
    QStyledItemDelegate,
    QSizePolicy,
    QApplication,
)
from PyQt6.QtCore import Qt, pyqtSignal, QEvent
from PyQt6.QtGui import QIcon, QColor

try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure

    MATPLOTLIB_AVAILABLE = True
except ImportError:

    class FigureCanvas(QWidget):
        pass

    class Figure(object):
        pass

    MATPLOTLIB_AVAILABLE = False
    logging.warning("matplotlib not found. PVT plot visualization will be disabled.")

try:
    import thermo
    from pint import UnitRegistry

    ureg = UnitRegistry()
    THERMO_AVAILABLE_FOR_UI = True
except ImportError:
    THERMO_AVAILABLE_FOR_UI = False
    logging.warning("thermo or pint library not found. UI auto-population will be disabled.")

PETROLEUM_COMPONENTS = sorted(
    [
        "CH4",
        "C2H6",
        "C3H8",
        "C4H10",
        "C5H12",
        "C6H14",
        "C7H16",
        "C8H18",
        "C9H20",
        "C10H22",
        "C11H24",
        "C12H26",
        "C13H28",
        "C14H30",
        "C15H32",
        "C16H34",
        "C17H36",
        "C18H38",
        "C19H40",
        "C20H42",
        "iC4H10",
        "iC5H12",
        "N2",
        "CO2",
        "H2S",
        "H2O",
    ]
)

from .pvt_table_editor import PVTTableEditorWidget
from core.unified_engine.physics.eos import CubicEOS, ReservoirFluid
from core.data_models import EOSModelParameters

PHYSICS_ENGINE_AVAILABLE = True

logger = logging.getLogger(__name__)


class ComponentDelegate(QStyledItemDelegate):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.completer_list = PETROLEUM_COMPONENTS

    def createEditor(self, parent: QWidget, option, index) -> QWidget:
        editor = QLineEdit(parent)
        completer = QCompleter(self.completer_list, editor)
        completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        completer.setFilterMode(Qt.MatchFlag.MatchContains)
        editor.setCompleter(completer)
        return editor


class PVTEditorDialog(QDialog):
    def __init__(
        self, initial_data: Optional[Dict[str, Any]] = None, parent: Optional[QWidget] = None
    ):
        super().__init__(parent)
        self.setMinimumSize(1200, 800)
        self.pvt_editors: Dict[str, PVTTableEditorWidget] = {}
        self.last_calculated_data: Optional[pd.DataFrame] = None

        self._setup_ui()
        self._connect_signals()

        if initial_data:
            self.load_data(initial_data)
        else:
            if hasattr(self, "comp_table"):
                self._add_eos_component("CO2")
                self._add_eos_component("CH4")
                self._add_eos_component("C10H22")

        self.retranslateUi()
        self.update_validation_status(
            self.tr("Ready. Define fluid model and click 'Run EOS' to plot."), "info"
        )

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.tabs = QTabWidget()
        self.pvt_editors["PVTO"] = PVTTableEditorWidget()
        self.pvt_editors["PVTG"] = PVTTableEditorWidget()
        self.pvt_editors["PVTW"] = PVTTableEditorWidget()
        self.tabs.addTab(self.pvt_editors["PVTO"], "")
        self.tabs.addTab(self.pvt_editors["PVTG"], "")
        self.tabs.addTab(self.pvt_editors["PVTW"], "")
        self._create_advanced_tab()
        splitter.addWidget(self.tabs)
        self.plot_widget = QWidget()
        plot_layout = QVBoxLayout(self.plot_widget)
        if MATPLOTLIB_AVAILABLE:
            self.fig = Figure(figsize=(5, 4), tight_layout=True)
            self.canvas = FigureCanvas(self.fig)
            self.ax1 = self.fig.add_subplot(111)
            plot_layout.addWidget(self.canvas)
        else:
            self.matplotlib_warning_label = QLabel()
            plot_layout.addWidget(self.matplotlib_warning_label)
        splitter.addWidget(self.plot_widget)
        splitter.setSizes([700, 500])
        main_layout.addWidget(splitter, 1)
        self.validation_label = QLabel()
        main_layout.addWidget(self.validation_label)
        self.button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        main_layout.addWidget(self.button_box)

    def _create_advanced_tab(self):
        adv_widget = QWidget()
        layout = QVBoxLayout(adv_widget)

        self.description_label = QLabel()
        self.description_label.setWordWrap(True)
        self.description_label.setStyleSheet(
            "background-color: #f0f8ff; border: 1px solid #d1e2ff; padding: 5px; border-radius: 3px;"
        )
        layout.addWidget(self.description_label)

        top_form_layout = QFormLayout()
        self.eos_combo = QComboBox()
        self.eos_label = QLabel()
        top_form_layout.addRow(self.eos_label, self.eos_combo)

        self.ref_temp_edit = QLineEdit("212.0")
        self.ref_temp_label = QLabel()
        top_form_layout.addRow(self.ref_temp_label, self.ref_temp_edit)
        layout.addLayout(top_form_layout)

        self.comp_group = QGroupBox()
        self.comp_group.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        comp_layout = QVBoxLayout(self.comp_group)
        self.comp_table = QTableWidget()
        self.comp_table.setItemDelegateForColumn(0, ComponentDelegate(self))
        self.comp_table.setColumnCount(7)
        header = self.comp_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(6, QHeaderView.ResizeMode.Stretch)

        comp_button_layout = QHBoxLayout()
        self.add_comp_btn = QPushButton(QIcon.fromTheme("list-add"), "")
        self.remove_comp_btn = QPushButton(QIcon.fromTheme("list-remove"), "")
        self.normalize_btn = QPushButton(QIcon.fromTheme("view-refresh"), "")
        comp_button_layout.addWidget(self.add_comp_btn)
        comp_button_layout.addWidget(self.remove_comp_btn)
        comp_button_layout.addStretch()
        comp_button_layout.addWidget(self.normalize_btn)
        comp_layout.addWidget(self.comp_table)
        comp_layout.addLayout(comp_button_layout)
        layout.addWidget(self.comp_group, 1)

        self.bip_group = QGroupBox()
        self.bip_group.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        bip_layout = QVBoxLayout(self.bip_group)
        self.bip_table = QTableWidget()
        self.bip_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)

        run_button_layout = QHBoxLayout()
        run_button_layout.addStretch()
        self.run_eos_btn = QPushButton()
        run_button_layout.addWidget(self.run_eos_btn)

        bip_layout.addWidget(self.bip_table, 1)
        bip_layout.addLayout(run_button_layout)
        layout.addWidget(self.bip_group, 1)

        self.tabs.addTab(adv_widget, "")

    def retranslateUi(self):
        self.setWindowTitle(self.tr("Detailed PVT Data Editor"))

        self.pvt_editors["PVTO"].table_name = self.tr("Oil PVT")
        self.pvt_editors["PVTO"].default_headers = [
            self.tr("Pressure (psia)"),
            self.tr("Rs (scf/STB)"),
            self.tr("Bo (RB/STB)"),
            self.tr("Oil Viscosity (cP)"),
        ]
        self.pvt_editors["PVTG"].table_name = self.tr("Gas PVT")
        self.pvt_editors["PVTG"].default_headers = [
            self.tr("Pressure (psia)"),
            self.tr("Bg (rcf/scf)"),
            self.tr("Gas Viscosity (cP)"),
        ]
        self.pvt_editors["PVTW"].table_name = self.tr("Water PVT")
        self.pvt_editors["PVTW"].default_headers = [
            self.tr("Ref Pressure (psia)"),
            self.tr("Bw (RB/STB)"),
            self.tr("Cw (1/psi)"),
            self.tr("Viscosity (cP)"),
            self.tr("Visc-b (1/psi)"),
        ]

        self.tabs.setTabText(0, self.tr("Oil (PVTO)"))
        self.tabs.setTabText(1, self.tr("Gas (PVTG)"))
        self.tabs.setTabText(2, self.tr("Water (PVTW)"))
        self.tabs.setTabText(3, self.tr("Advanced (EOS)"))

        if not MATPLOTLIB_AVAILABLE:
            self.matplotlib_warning_label.setText(
                self.tr("Matplotlib not installed. Plotting is disabled.")
            )

        self.validation_label.setText(self.tr("Ready."))

        self.description_label.setText(
            self.tr(
                "<b>EOS Workflow:</b>\n"
                "1. Define components, mole fractions, and reservoir conditions.\n"
                "2. Manually edit Binary Interaction Parameters (k_ij) as needed.\n"
                "3. Click <b>'Run EOS & Plot'</b> to calculate properties and update the plot.\n"
                "4. Compare the calculated lines against experimental data points."
            )
        )

        # Temporarily block signals to avoid triggering events on item change
        self.eos_combo.blockSignals(True)
        current_text = self.eos_combo.currentText()
        self.eos_combo.clear()
        self.eos_combo.addItems([self.tr("Peng-Robinson"), self.tr("Soave-Redlich-Kwong")])
        if current_text in [self.tr("Peng-Robinson"), self.tr("Soave-Redlich-Kwong")]:
            self.eos_combo.setCurrentText(current_text)
        self.eos_combo.blockSignals(False)

        self.eos_label.setText(self.tr("Equation of State (EOS):"))
        self.ref_temp_label.setText(self.tr("Reservoir Temperature (°F):"))
        self.comp_group.setTitle(
            self.tr("Component Properties (Enter chemical formula, e.g., CH4, C2H6, CO2)")
        )

        headers = [
            self.tr("Component"),
            self.tr("Mol Frac (zi)"),
            self.tr("MW"),
            self.tr("Tc (°R)"),
            self.tr("Pc (psia)"),
            self.tr("Acentric Factor"),
            self.tr("Volume Shift"),
        ]
        self.comp_table.setHorizontalHeaderLabels(headers)

        self.add_comp_btn.setText(self.tr("Add Blank Row"))
        self.remove_comp_btn.setText(self.tr("Remove Selected"))
        self.normalize_btn.setText(self.tr("Normalize Fractions"))
        self.bip_group.setTitle(self.tr("Binary Interaction Parameters (k_ij)"))
        self.run_eos_btn.setText(self.tr("Run EOS & Plot"))

        # Update plot with translated text
        self.update_plot(self.last_calculated_data)

    def changeEvent(self, event: QEvent):
        if event.type() == QEvent.Type.LanguageChange:
            self.retranslateUi()
        super().changeEvent(event)

    def _connect_signals(self):
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        self.tabs.currentChanged.connect(lambda: self.update_plot(self.last_calculated_data))
        for editor in self.pvt_editors.values():
            if isinstance(editor, PVTTableEditorWidget):
                editor.data_changed.connect(lambda: self.update_plot(self.last_calculated_data))

        if hasattr(self, "add_comp_btn"):
            self.add_comp_btn.clicked.connect(self._add_blank_component_row)
            self.remove_comp_btn.clicked.connect(self._remove_selected_eos_component)
            self.comp_table.itemChanged.connect(self._on_component_data_changed)
            self.bip_table.itemChanged.connect(self._on_bip_changed)
            self.normalize_btn.clicked.connect(self._normalize_mole_fractions)
            self.run_eos_btn.clicked.connect(self._run_and_plot_full_pvt_range)

    def update_plot(self, calculated_data: Optional[pd.DataFrame] = None):
        if not MATPLOTLIB_AVAILABLE:
            return

        current_tab_text = self.tabs.tabText(self.tabs.currentIndex())
        self.plot_widget.setVisible(
            "PVT" in current_tab_text or self.tr("Advanced") in current_tab_text
        )  # Check for translated "Advanced" too
        self.fig.clear()
        self.ax1 = self.fig.add_subplot(111)
        ax2 = None
        plots = []

        if self.tr("Oil") in current_tab_text:
            self.ax1.set_title(self.tr("Oil PVT Properties vs. Pressure"))
            self.ax1.set_xlabel(self.tr("Pressure (psia)"))
            self.ax1.grid(True, which="both", linestyle="--")

            current_editor = self.pvt_editors.get("PVTO")
            data = (
                current_editor.get_data_as_numpy()
                if isinstance(current_editor, PVTTableEditorWidget)
                else np.empty((0, 0))
            )

            if data.shape[0] > 0 and data.shape[1] >= 4:
                df = (
                    pd.DataFrame(data, columns=["pressure", "rs", "bo", "visc"])
                    .dropna()
                    .sort_values("pressure")
                )
                ax2 = self.ax1.twinx()
                p1 = self.ax1.plot(
                    df["pressure"], df["bo"], "o", color="green", label=self.tr("Bo (Exp.)")
                )
                p2 = ax2.plot(
                    df["pressure"],
                    df["rs"],
                    "s",
                    mfc="none",
                    color="blue",
                    label=self.tr("Rs (Exp.)"),
                )
                p3 = ax2.plot(
                    df["pressure"],
                    df["visc"],
                    "^",
                    mfc="none",
                    color="red",
                    label=self.tr("Viscosity (Exp.)"),
                )
                plots.extend(p1 + p2 + p3)
                self.ax1.set_ylabel(self.tr("Oil FVF (Bo, RB/STB)"), color="green")
                self.ax1.tick_params(axis="y", labelcolor="green")
                ax2.set_ylabel(self.tr("Rs (scf/STB) / Viscosity (cP)"))

            if calculated_data is not None and not calculated_data.empty:
                calc_df = calculated_data.sort_values("pressure")
                if "oil_fvf_rb_stb" in calc_df.columns and calc_df["oil_fvf_rb_stb"].notna().any():
                    plots.extend(
                        self.ax1.plot(
                            calc_df["pressure"],
                            calc_df["oil_fvf_rb_stb"],
                            "--",
                            color="darkgreen",
                            label=self.tr("Bo (Calc.)"),
                        )
                    )
                if (
                    "oil_viscosity_cp" in calc_df.columns
                    and calc_df["oil_viscosity_cp"].notna().any()
                ):
                    if ax2 is None:
                        ax2 = self.ax1.twinx()
                        ax2.set_ylabel(self.tr("Viscosity (cP)"), color="red")
                        ax2.tick_params(axis="y", labelcolor="red")
                    plots.extend(
                        ax2.plot(
                            calc_df["pressure"],
                            calc_df["oil_viscosity_cp"],
                            ":",
                            color="darkred",
                            label=self.tr("Viscosity (Calc.)"),
                        )
                    )

        elif self.tr("Gas") in current_tab_text:
            self.ax1.set_title(self.tr("Gas PVT Properties vs. Pressure"))
            self.ax1.set_xlabel(self.tr("Pressure (psia)"))
            self.ax1.grid(True, which="both", linestyle="--")

            current_editor = self.pvt_editors.get("PVTG")
            data = (
                current_editor.get_data_as_numpy()
                if isinstance(current_editor, PVTTableEditorWidget)
                else np.empty((0, 0))
            )

            if data.shape[0] > 0 and data.shape[1] >= 3:
                df = (
                    pd.DataFrame(data, columns=["pressure", "bg", "visc"])
                    .dropna()
                    .sort_values("pressure")
                )
                ax2 = self.ax1.twinx()
                p1 = self.ax1.plot(
                    df["pressure"], df["bg"], "o", color="purple", label=self.tr("Bg (Exp.)")
                )
                p2 = ax2.plot(
                    df["pressure"],
                    df["visc"],
                    "^",
                    mfc="none",
                    color="orange",
                    label=self.tr("Viscosity (Exp.)"),
                )
                plots.extend(p1 + p2)
                self.ax1.set_ylabel(self.tr("Gas FVF (Bg, rcf/scf)"), color="purple")
                self.ax1.tick_params(axis="y", labelcolor="purple")
                ax2.set_ylabel(self.tr("Viscosity (cP)"), color="orange")
                ax2.tick_params(axis="y", labelcolor="orange")

            if calculated_data is not None and not calculated_data.empty:
                calc_df = calculated_data.sort_values("pressure")
                if (
                    "gas_fvf_rcf_scf" in calc_df.columns
                    and calc_df["gas_fvf_rcf_scf"].notna().any()
                ):
                    plots.extend(
                        self.ax1.plot(
                            calc_df["pressure"],
                            calc_df["gas_fvf_rcf_scf"],
                            "--",
                            color="darkmagenta",
                            label=self.tr("Bg (Calc.)"),
                        )
                    )
                if (
                    "gas_viscosity_cp" in calc_df.columns
                    and calc_df["gas_viscosity_cp"].notna().any()
                ):
                    if ax2 is None:
                        ax2 = self.ax1.twinx()
                        ax2.set_ylabel(self.tr("Viscosity (cP)"), color="orange")
                        ax2.tick_params(axis="y", labelcolor="orange")
                    plots.extend(
                        ax2.plot(
                            calc_df["pressure"],
                            calc_df["gas_viscosity_cp"],
                            ":",
                            color="darkorange",
                            label=self.tr("Viscosity (Calc.)"),
                        )
                    )

        else:
            self.ax1.set_title(self.tr("No Data to Display"))
            self.ax1.text(
                0.5,
                0.5,
                self.tr("Select Oil or Gas PVT tab to see a plot"),
                ha="center",
                va="center",
                fontsize=12,
                color="gray",
            )

        if plots:
            labels = [p.get_label() for p in plots]
            self.ax1.legend(plots, labels, loc="best")

        try:
            self.fig.tight_layout(pad=2.0)
        except Exception:
            pass
        self.canvas.draw()

    def _run_and_plot_full_pvt_range(self):
        eos_params = self._get_eos_model_from_ui(show_errors=True)
        if not eos_params:
            self.update_plot(calculated_data=None)
            self.update_validation_status(
                self.tr("Could not run calculation. Please check EOS data."), "error"
            )
            return

        current_tab_text = self.tabs.tabText(self.tabs.currentIndex())
        target_tab_text = None

        if self.tr("Advanced") in current_tab_text:
            msg_box = QMessageBox(self)
            msg_box.setWindowTitle(self.tr("Select Plot Target"))
            msg_box.setText(self.tr("Which PVT data should be used for the calculation and plot?"))
            msg_box.setIcon(QMessageBox.Icon.Question)
            oil_button = msg_box.addButton(self.tr("Oil (PVTO)"), QMessageBox.ButtonRole.ActionRole)
            gas_button = msg_box.addButton(self.tr("Gas (PVTG)"), QMessageBox.ButtonRole.ActionRole)
            msg_box.addButton(QMessageBox.StandardButton.Cancel)
            msg_box.exec()

            clicked_button = msg_box.clickedButton()
            if clicked_button == oil_button:
                target_tab_text = self.tr("Oil (PVTO)")
            elif clicked_button == gas_button:
                target_tab_text = self.tr("Gas (PVTG)")
            else:
                return
        else:
            target_tab_text = current_tab_text

        pvt_editor_key = None
        headers = []
        property_to_check = None

        if self.tr("Oil") in target_tab_text:
            pvt_editor_key = "PVTO"
            headers = ["pressure", "rs", "bo", "visc"]
            property_to_check = "oil_fvf_rb_stb"
        elif self.tr("Gas") in target_tab_text:
            pvt_editor_key = "PVTG"
            headers = ["pressure", "bg", "visc"]
            property_to_check = "gas_fvf_rcf_scf"
        else:
            QMessageBox.information(
                self,
                self.tr("No Plot"),
                self.tr("Calculations can only be run for 'Oil (PVTO)' or 'Gas (PVTG)' data."),
            )
            return

        pvt_editor = self.pvt_editors.get(pvt_editor_key)
        pressures = None
        if isinstance(pvt_editor, PVTTableEditorWidget):
            exp_data_np = pvt_editor.get_data_as_numpy()
            if exp_data_np.shape[0] > 0:
                exp_df = pd.DataFrame(exp_data_np, columns=headers).dropna()
                if not exp_df.empty:
                    pressures = exp_df["pressure"].values

        if pressures is None:
            QMessageBox.warning(
                self,
                self.tr("No Experimental Data"),
                self.tr(
                    "There is no experimental data to define a pressure range. Calculation will run over a default range."
                ),
            )
            pressures = np.linspace(14.7, 5000, 20)

        try:
            T_test = float(self.ref_temp_edit.text())
            if PHYSICS_ENGINE_AVAILABLE and CubicEOS:
                # Use ReservoirFluid wrapper which properly converts EOSModelParameters to EOSParameters
                model = ReservoirFluid(eos_params).eos_model
            else:
                # Fallback when physics engine is not available
                logging.warning("Physics engine not available, skipping PVT calculation")
                return

            self.update_validation_status(self.tr("Calculating PVT properties..."), "info")
            QApplication.processEvents()

            calculated_results = [
                model.calculate_properties(p, T_test) | {"pressure": p} for p in pressures
            ]
            self.last_calculated_data = pd.DataFrame(calculated_results)

            if (
                property_to_check not in self.last_calculated_data
                or self.last_calculated_data[property_to_check].notna().sum() == 0
            ):
                QMessageBox.warning(
                    self,
                    self.tr("Calculation Warning"),
                    self.tr(
                        "The EOS calculation ran successfully, but it did not produce any valid properties for the selected phase (Oil or Gas) "
                        "at the given pressure and temperature conditions.\n\n"
                        "This often means the model is predicting a different phase (e.g., gas instead of liquid). Please check:\n"
                        "1. The fluid composition and mole fractions.\n"
                        "2. The reservoir temperature.\n"
                        "3. The Binary Interaction Parameters (k_ij), as default values of zero may be incorrect for your mixture."
                    ),
                )
                self.update_validation_status(
                    self.tr(
                        "Calculation complete, but no valid properties were generated for the target phase."
                    ),
                    "warning",
                )
            else:
                self.update_validation_status(
                    self.tr("PVT calculation successful. Plot updated."), "ok"
                )

            for i in range(self.tabs.count()):
                if self.tabs.tabText(i) == target_tab_text:
                    self.tabs.setCurrentIndex(i)
                    break

            self.update_plot(self.last_calculated_data)

        except Exception as e:
            QMessageBox.critical(
                self,
                self.tr("Calculation Error"),
                self.tr(
                    "An error occurred while calculating PVT properties for the plot: {e}"
                ).format(e=e),
            )
            self.update_validation_status(
                self.tr("Error during PVT calculation: {e}").format(e=e), "error"
            )

    def get_data(self) -> Dict[str, Any]:
        output = {}
        for name, editor in self.pvt_editors.items():
            if isinstance(editor, PVTTableEditorWidget):
                output[name] = editor.get_data_as_numpy()
        eos_params = self._get_eos_model_from_ui(show_errors=False)
        if isinstance(eos_params, EOSModelParameters):
            output["eos_model"] = eos_params
        return output

    def _set_bip_table_data(self, bips: np.ndarray):
        self.bip_table.blockSignals(True)
        try:
            for r in range(min(bips.shape[0], self.bip_table.rowCount())):
                for c in range(min(bips.shape[1], self.bip_table.columnCount())):
                    if self.bip_table.item(r, c):
                        self.bip_table.item(r, c).setText(f"{bips[r, c]:.5f}")
        finally:
            self.bip_table.blockSignals(False)
        if self.bip_table.rowCount() > 1 and self.bip_table.columnCount() > 1:
            if self.bip_table.item(0, 1):
                self._on_bip_changed(self.bip_table.item(0, 1))

    def _add_blank_component_row(self):
        row_pos = self.comp_table.rowCount()
        self.comp_table.insertRow(row_pos)
        self.comp_table.blockSignals(True)
        try:
            self.comp_table.setItem(row_pos, 0, QTableWidgetItem(""))
            for col in range(1, self.comp_table.columnCount()):
                self.comp_table.setItem(row_pos, col, QTableWidgetItem("0.0"))
        finally:
            self.comp_table.blockSignals(False)
        self.comp_table.editItem(self.comp_table.item(row_pos, 0))

    def _add_eos_component(self, name: str):
        row_pos = self.comp_table.rowCount()
        self.comp_table.insertRow(row_pos)
        self.comp_table.blockSignals(True)
        try:
            name_item = QTableWidgetItem(name)
            self.comp_table.setItem(row_pos, 0, name_item)
            for col in range(1, self.comp_table.columnCount()):
                self.comp_table.setItem(row_pos, col, QTableWidgetItem("0.0"))
        finally:
            self.comp_table.blockSignals(False)
        self._validate_and_populate_row(name_item)
        self._update_bip_table()

    def _remove_selected_eos_component(self):
        selected_rows = sorted(
            list(set(item.row() for item in self.comp_table.selectedItems())), reverse=True
        )
        if not selected_rows:
            return
        for row in selected_rows:
            self.comp_table.removeRow(row)
        self._update_bip_table()

    def _on_component_data_changed(self, item: QTableWidgetItem):
        if item.column() == 0:
            self._validate_and_populate_row(item)
            self._update_bip_table()

    def _on_bip_changed(self, item: QTableWidgetItem):
        r, c = item.row(), item.column()
        if r != c and c > r:
            self.bip_table.blockSignals(True)
            try:
                mirror_item = self.bip_table.item(c, r)
                if mirror_item:
                    mirror_item.setText(item.text())
            finally:
                self.bip_table.blockSignals(False)

    def _validate_and_populate_row(self, name_item: QTableWidgetItem):
        input_name = name_item.text().strip()
        row = name_item.row()
        if not input_name:
            self._clear_component_row(row)
            return
        if not THERMO_AVAILABLE_FOR_UI:
            self.update_validation_status(
                self.tr("Cannot validate: thermo library not available."), "error"
            )
            return
        try:
            chem = thermo.Chemical(input_name)
            if chem.MW is None:
                raise ValueError("Component not found")
            self._populate_component_row(row, input_name, chem_obj=chem)
        except Exception:
            self._clear_component_row(row)
            self.update_validation_status(
                self.tr("'{input_name}' is not a recognized component.").format(
                    input_name=input_name
                ),
                "error",
            )

    def _clear_component_row(self, row: int):
        self.comp_table.blockSignals(True)
        try:
            for col in range(2, self.comp_table.columnCount()):
                item = self.comp_table.item(row, col)
                if item:
                    item.setText("0.0")
        finally:
            self.comp_table.blockSignals(False)

    def _populate_component_row(self, row: int, name: str, chem_obj=None):
        if not THERMO_AVAILABLE_FOR_UI:
            return
        self.comp_table.blockSignals(True)
        try:
            chem = chem_obj or thermo.Chemical(name)
            self.comp_table.setItem(row, 2, QTableWidgetItem(f"{chem.MW:.4f}"))
            self.comp_table.setItem(
                row, 3, QTableWidgetItem(f"{ureg.Quantity(chem.Tc, 'K').to('degR').magnitude:.2f}")
            )
            self.comp_table.setItem(
                row, 4, QTableWidgetItem(f"{ureg.Quantity(chem.Pc, 'Pa').to('psi').magnitude:.2f}")
            )
            self.comp_table.setItem(row, 5, QTableWidgetItem(f"{chem.omega:.4f}"))
            self.update_validation_status(self.tr("Validated '{name}'.").format(name=name), "ok")
        except Exception as e:
            logger.warning(f"Could not auto-populate properties for '{name}': {e}")
        finally:
            self.comp_table.blockSignals(False)

    def _update_bip_table(self):
        self.bip_table.blockSignals(True)
        try:
            components = [
                self.comp_table.item(r, 0).text()
                for r in range(self.comp_table.rowCount())
                if self.comp_table.item(r, 0) and self.comp_table.item(r, 0).text()
            ]
            old_bips = self._get_table_data_as_numpy(self.bip_table)
            old_headers = (
                [
                    self.bip_table.horizontalHeaderItem(i).text()
                    for i in range(self.bip_table.columnCount())
                ]
                if self.bip_table.columnCount() > 0
                else []
            )

            self.bip_table.clearContents()
            self.bip_table.setRowCount(len(components))
            self.bip_table.setColumnCount(len(components))
            self.bip_table.setHorizontalHeaderLabels(components)
            self.bip_table.setVerticalHeaderLabels(components)
            gray_color = QColor(240, 240, 240)

            header = self.bip_table.horizontalHeader()
            for i in range(len(components)):
                header.setSectionResizeMode(i, QHeaderView.ResizeMode.Stretch)

            for r, r_name in enumerate(components):
                for c, c_name in enumerate(components):
                    item = QTableWidgetItem("0.0")
                    if r_name in old_headers and c_name in old_headers:
                        try:
                            old_r_idx = old_headers.index(r_name)
                            old_c_idx = old_headers.index(c_name)
                            if not np.isnan(old_bips[old_r_idx, old_c_idx]):
                                item.setText(f"{old_bips[old_r_idx, old_c_idx]:.5f}")
                        except (ValueError, IndexError):
                            pass

                    if c <= r:
                        item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                        item.setBackground(gray_color)
                        if c < r and self.bip_table.item(c, r):
                            item.setText(self.bip_table.item(c, r).text())
                    self.bip_table.setItem(r, c, item)
        finally:
            self.bip_table.blockSignals(False)

    def _normalize_mole_fractions(self):
        self.comp_table.blockSignals(True)
        try:
            fractions = []
            for r in range(self.comp_table.rowCount()):
                item = self.comp_table.item(r, 1)
                if item and item.text().strip():
                    fractions.append(float(item.text().strip()))
                else:
                    fractions.append(0.0)

            total = sum(fractions)
            if total > 1e-9:
                for r in range(len(fractions)):
                    self.comp_table.item(r, 1).setText(f"{(fractions[r] / total):.6f}")
                self.update_validation_status(self.tr("Mole fractions normalized."), "ok")
        except (ValueError, AttributeError):
            self.update_validation_status(self.tr("Invalid mole fraction value detected."), "error")
        finally:
            self.comp_table.blockSignals(False)

    def _get_eos_model_from_ui(self, show_errors=True) -> Optional[EOSModelParameters]:
        try:
            self._normalize_mole_fractions()

            rows = self.comp_table.rowCount()
            cols = self.comp_table.columnCount()
            if rows == 0:
                if show_errors:
                    QMessageBox.warning(
                        self,
                        self.tr("No Components"),
                        self.tr("Please add at least one component to the EOS model."),
                    )
                return None

            comp_props = np.empty((rows, cols), dtype=object)
            for r in range(rows):
                item = self.comp_table.item(r, 0)
                comp_name = item.text().strip() if item else ""
                if not comp_name:
                    raise ValueError(
                        self.tr("Component name in row {row_num} cannot be empty.").format(
                            row_num=r + 1
                        )
                    )
                comp_props[r, 0] = comp_name

                for c in range(1, cols):
                    item = self.comp_table.item(r, c)
                    text_val = item.text().strip() if item and item.text() else "0.0"
                    comp_props[r, c] = text_val

            bip_data = self._get_table_data_as_numpy(self.bip_table)
            bip_data[np.isnan(bip_data)] = 0.0

            eos_type = ""
            current_combo_text = self.eos_combo.currentText()
            if current_combo_text == self.tr("Peng-Robinson"):
                eos_type = "Peng-Robinson"
            elif current_combo_text == self.tr("Soave-Redlich-Kwong"):
                eos_type = "Soave-Redlich-Kwong"
            else:  # Fallback for when language is switched and text might not match
                eos_type = "Peng-Robinson"

            model_params = EOSModelParameters(
                eos_type=eos_type,
                component_properties=comp_props,
                binary_interaction_coeffs=bip_data,
            )
            return model_params
        except (ValueError, TypeError) as e:
            if show_errors:
                QMessageBox.critical(
                    self,
                    self.tr("EOS Data Error"),
                    self.tr("Could not parse EOS data.\nError: {e}").format(e=e),
                )
            return None
        except Exception as e:
            if show_errors:
                QMessageBox.critical(
                    self,
                    self.tr("Unexpected EOS Error"),
                    self.tr("An unexpected error occurred.\nError: {e}").format(e=e),
                )
            return None

    def update_validation_status(self, message: str, level: str):
        self.validation_label.setText(f"{self.tr('Status')}: {message}")
        color = {"ok": "#2ECC71", "warning": "#F39C12", "error": "#E74C3C", "info": "#3498DB"}.get(
            level, "black"
        )
        self.validation_label.setStyleSheet(f"color: {color}; font-weight: bold;")

    def _get_table_data_as_numpy(self, table: QTableWidget, dtype=float) -> np.ndarray:
        data = np.full((table.rowCount(), table.columnCount()), np.nan, dtype=dtype)
        for r in range(table.rowCount()):
            for c in range(table.columnCount()):
                item = table.item(r, c)
                if item and item.text().strip():
                    try:
                        data[r, c] = dtype(item.text().strip())
                    except (ValueError, TypeError):
                        pass
        return data

    def _get_table_data_as_object(self, table: QTableWidget) -> np.ndarray:
        data = np.empty((table.rowCount(), table.columnCount()), dtype=object)
        for r in range(table.rowCount()):
            for c in range(table.columnCount()):
                item = table.item(r, c)
                data[r, c] = item.text() if item else ""
        return data

    def load_data(self, data: Dict[str, Any]):
        for name, editor in self.pvt_editors.items():
            if (
                name in data
                and isinstance(data[name], np.ndarray)
                and isinstance(editor, PVTTableEditorWidget)
            ):
                editor.set_full_data(data[name])

        eos_model = data.get("eos_model")
        if isinstance(eos_model, EOSModelParameters):
            eos_type_map = {
                "Peng-Robinson": self.tr("Peng-Robinson"),
                "Soave-Redlich-Kwong": self.tr("Soave-Redlich-Kwong"),
            }
            self.eos_combo.setCurrentText(eos_type_map.get(eos_model.eos_type, ""))

            self.comp_table.blockSignals(True)
            try:
                comp_props = eos_model.component_properties
                self.comp_table.setRowCount(comp_props.shape[0])
                for r in range(comp_props.shape[0]):
                    for c in range(comp_props.shape[1]):
                        self.comp_table.setItem(r, c, QTableWidgetItem(str(comp_props[r, c])))
            finally:
                self.comp_table.blockSignals(False)

            self._update_bip_table()
            self._set_bip_table_data(eos_model.binary_interaction_coeffs)
