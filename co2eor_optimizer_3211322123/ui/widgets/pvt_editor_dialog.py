import logging
from typing import Optional, Dict, Any

import numpy as np
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QPushButton, QTabWidget, QMessageBox, QLineEdit, QComboBox,
    QDialogButtonBox, QWidget, QSplitter, QFormLayout, QTableWidget,
    QTableWidgetItem, QHeaderView
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QIcon

# This feature requires matplotlib
try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    class FigureCanvas(QWidget): pass
    class Figure(object): pass
    MATPLOTLIB_AVAILABLE = False
    logging.warning("matplotlib not found. PVT plot visualization will be disabled.")

try:
    # Assuming this widget is now in a sub-folder like 'widgets' relative to the dialog
    from .pvt_table_editor import PVTTableEditorWidget
    from co2eor_optimizer.core.data_models import EOSModelParameters
except ImportError:
    class PVTTableEditorWidget(QWidget): pass
    class EOSModelParameters: pass
    logging.critical("PVTEditorDialog: Failed to import dependencies.")

logger = logging.getLogger(__name__)

class PVTEditorDialog(QDialog):
    """
    A comprehensive dialog for editing detailed PVT data, including black oil tables
    and advanced Equation of State (EOS) compositional models, with real-time visualization.
    """
    def __init__(self, initial_data: Optional[Dict[str, Any]] = None, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("Detailed PVT Data Editor")
        self.setMinimumSize(1100, 700)

        self.pvt_editors: Dict[str, PVTTableEditorWidget] = {}
        self._setup_ui()
        self._connect_signals()

        if initial_data:
            self.load_data(initial_data)
        else:
            # Add some default components for a new EOS model
            self._add_eos_component("CO2")
            self._add_eos_component("C1")
            self._add_eos_component("C10")

        self.update_plot()
        self.update_validation_status("Ready.", "info")

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)

        top_group = QGroupBox("Reference & Global Properties")
        top_layout = QHBoxLayout(top_group)
        ref_form_layout = QFormLayout()
        self.ref_pressure_edit = QLineEdit("3000.0")
        self.ref_temp_edit = QLineEdit("212.0")
        ref_form_layout.addRow("Reference Pressure (psia):", self.ref_pressure_edit)
        ref_form_layout.addRow("Reference Temperature (°F):", self.ref_temp_edit)
        top_layout.addLayout(ref_form_layout)
        main_layout.addWidget(top_group)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        self.tabs = QTabWidget()
        self.pvt_editors['PVTO'] = PVTTableEditorWidget("Oil PVT", ["Pressure (psia)", "Rs (scf/STB)", "Bo (RB/STB)", "Oil Viscosity (cP)"])
        self.pvt_editors['PVTG'] = PVTTableEditorWidget("Gas PVT", ["Pressure (psia)", "Bg (RB/MSCF)", "Gas Viscosity (cP)"])
        self.pvt_editors['PVTW'] = PVTTableEditorWidget("Water PVT", ["Ref Pressure (psia)", "Bw (RB/STB)", "Cw (1/psi)", "Viscosity (cP)", "Visc-b (1/psi)"])
        self.tabs.addTab(self.pvt_editors['PVTO'], "Oil (PVTO)")
        self.tabs.addTab(self.pvt_editors['PVTG'], "Gas (PVTG)")
        self.tabs.addTab(self.pvt_editors['PVTW'], "Water (PVTW)")
        self._create_advanced_tab() # INTEGRATION: Add EOS tab
        splitter.addWidget(self.tabs)
        
        plot_widget = QWidget()
        plot_layout = QVBoxLayout(plot_widget)
        if MATPLOTLIB_AVAILABLE:
            self.fig = Figure(figsize=(5, 4), tight_layout=True)
            self.canvas = FigureCanvas(self.fig)
            self.ax1 = self.fig.add_subplot(111)
            plot_layout.addWidget(self.canvas)
        else:
            plot_layout.addWidget(QLabel("Matplotlib not installed. Plotting is disabled."))
        splitter.addWidget(plot_widget)
        splitter.setSizes([600, 500])
        main_layout.addWidget(splitter, 1)

        self.validation_label = QLabel("Ready.")
        main_layout.addWidget(self.validation_label)
        self.button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        main_layout.addWidget(self.button_box)

    def _create_advanced_tab(self):
        """Creates the tab for editing EOS model parameters."""
        adv_widget = QWidget()
        layout = QVBoxLayout(adv_widget)
        
        eos_selection_layout = QHBoxLayout()
        eos_selection_layout.addWidget(QLabel("Equation of State (EOS):"))
        self.eos_combo = QComboBox()
        self.eos_combo.addItems(["Peng-Robinson", "Soave-Redlich-Kwong"])
        eos_selection_layout.addWidget(self.eos_combo)
        eos_selection_layout.addStretch()
        layout.addLayout(eos_selection_layout)

        # Component Properties Table
        comp_group = QGroupBox("Component Properties")
        comp_layout = QVBoxLayout(comp_group)
        self.comp_table = QTableWidget()
        self.comp_table.setColumnCount(7)
        self.comp_table.setHorizontalHeaderLabels(["Component", "Mol Frac (zi)", "MW", "Tc (°R)", "Pc (psia)", "Acentric Factor", "Volume Shift"])
        self.comp_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        comp_button_layout = QHBoxLayout()
        self.add_comp_btn = QPushButton(QIcon.fromTheme("list-add"), "Add Component")
        self.remove_comp_btn = QPushButton(QIcon.fromTheme("list-remove"), "Remove Selected")
        comp_button_layout.addWidget(self.add_comp_btn)
        comp_button_layout.addWidget(self.remove_comp_btn)
        comp_button_layout.addStretch()
        comp_layout.addWidget(self.comp_table)
        comp_layout.addLayout(comp_button_layout)
        layout.addWidget(comp_group)
        
        # Binary Interaction Parameters Table
        bip_group = QGroupBox("Binary Interaction Parameters (k_ij)")
        bip_layout = QVBoxLayout(bip_group)
        self.bip_table = QTableWidget()
        self.bip_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.bip_table.verticalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        bip_layout.addWidget(self.bip_table)
        layout.addWidget(bip_group)

        self.tabs.addTab(adv_widget, "Advanced (EOS)")

    def _connect_signals(self):
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        self.tabs.currentChanged.connect(self.update_plot)
        for editor in self.pvt_editors.values():
            editor.data_changed.connect(self.update_plot)
            editor.validation_status.connect(self.update_validation_status)
        
        # EOS tab signals
        self.add_comp_btn.clicked.connect(lambda: self._add_eos_component())
        self.remove_comp_btn.clicked.connect(self._remove_selected_eos_component)
        self.comp_table.itemChanged.connect(self._on_component_name_changed)

    def _add_eos_component(self, name: str = ""):
        row_pos = self.comp_table.rowCount()
        self.comp_table.insertRow(row_pos)
        self.comp_table.setItem(row_pos, 0, QTableWidgetItem(name or f"NewComp-{row_pos+1}"))
        # Populate with zeros, user to fill in
        for col in range(1, self.comp_table.columnCount()):
            self.comp_table.setItem(row_pos, col, QTableWidgetItem("0.0"))
        self._update_bip_table()

    def _remove_selected_eos_component(self):
        selected_rows = sorted(list(set(item.row() for item in self.comp_table.selectedItems())), reverse=True)
        if not selected_rows:
            QMessageBox.warning(self, "No Selection", "Please select one or more components to remove.")
            return
        for row in selected_rows:
            self.comp_table.removeRow(row)
        self._update_bip_table()

    def _on_component_name_changed(self, item: QTableWidgetItem):
        # If a component name in the first column changes, update the BIP table headers
        if item.column() == 0:
            self._update_bip_table()

    def _update_bip_table(self):
        """Syncs the BIP table with the component table."""
        components = [self.comp_table.item(r, 0).text() for r in range(self.comp_table.rowCount()) if self.comp_table.item(r, 0)]
        self.bip_table.setRowCount(len(components))
        self.bip_table.setColumnCount(len(components))
        self.bip_table.setHorizontalHeaderLabels(components)
        self.bip_table.setVerticalHeaderLabels(components)
        for r in range(len(components)):
            for c in range(len(components)):
                if self.bip_table.item(r, c) is None:
                    item = QTableWidgetItem("0.0")
                    if r == c:
                        item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                        item.setToolTip("Diagonal elements must be zero.")
                    self.bip_table.setItem(r, c, item)

    def update_plot(self):
        if not MATPLOTLIB_AVAILABLE:
            return

        self.fig.clear()
        self.ax1 = self.fig.add_subplot(111)
        current_tab_text = self.tabs.tabText(self.tabs.currentIndex())

        # Only plot for the oil PVT tab
        if 'Oil' not in current_tab_text:
            self.ax1.set_title(f"No Plot Available for {current_tab_text}")
            self.ax1.set_xticks([]); self.ax1.set_yticks([])
            self.canvas.draw(); return

        oil_editor = self.pvt_editors['PVTO']
        data = oil_editor.get_data_as_numpy()
        valid_data = data[~np.isnan(data).any(axis=1)]
        
        self.ax1.set_title("Oil PVT Properties vs. Pressure")
        self.ax1.set_xlabel("Pressure (psia)")
        self.ax1.grid(True, linestyle='--', alpha=0.7)

        if valid_data.shape[0] > 1 and valid_data.shape[1] >= 4:
            valid_data = valid_data[valid_data[:, 0].argsort()] # Sort by pressure
            pressure, rs, bo, visc = valid_data[:, 0], valid_data[:, 1], valid_data[:, 2], valid_data[:, 3]
            
            # Plot Bo on primary y-axis
            self.ax1.set_ylabel("Bo (RB/STB)", color='g')
            self.ax1.plot(pressure, bo, 'o-', color='g', label='Bo (RB/STB)')
            self.ax1.tick_params(axis='y', labelcolor='g')
            
            # Create a secondary y-axis for Rs and Viscosity
            ax2 = self.ax1.twinx()
            ax2.set_ylabel("Rs / Viscosity") 
            ax2.plot(pressure, rs, 's--', color='b', label='Rs (scf/STB)')
            ax2.plot(pressure, visc, '^:', color='r', label='Oil Viscosity (cP)')
            
            # Combine legends
            self.ax1.legend(loc='upper left')
            ax2.legend(loc='upper right')
        else:
            self.ax1.text(0.5, 0.5, 'Enter at least two valid rows to see plot.', 
                          horizontalalignment='center', verticalalignment='center',
                          transform=self.ax1.transAxes)

        self.fig.tight_layout()
        self.canvas.draw()

    def update_validation_status(self, message: str, level: str):
        self.validation_label.setText(f"Status: {message}")
        color = {"ok": "#2ECC71", "warning": "#F39C12", "error": "#E74C3C", "info": "#3498DB"}.get(level, "black")
        self.validation_label.setStyleSheet(f"color: {color}; font-weight: bold;")
        
    def get_data(self) -> Dict[str, Any]:
        """Collects all data from the dialog into a structured dictionary."""
        try: ref_pres = float(self.ref_pressure_edit.text())
        except (ValueError, TypeError): ref_pres = 0.0
        try: ref_temp = float(self.ref_temp_edit.text())
        except (ValueError, TypeError): ref_temp = 0.0

        output = {'ref_pressure': ref_pres, 'ref_temperature': ref_temp}
        for name, editor in self.pvt_editors.items():
            output[name] = editor.get_data_as_numpy()
            
        # --- INTEGRATION: Collect data from the EOS tab ---
        try:
            comp_rows, comp_cols = self.comp_table.rowCount(), self.comp_table.columnCount()
            if comp_rows > 0:
                # Use object dtype to handle strings (names) and numbers
                comp_props = np.empty((comp_rows, comp_cols), dtype=object)
                for r in range(comp_rows):
                    for c in range(comp_cols):
                        comp_props[r, c] = self.comp_table.item(r, c).text() if self.comp_table.item(r, c) else ""

                bip_size = self.bip_table.rowCount()
                bips = np.zeros((bip_size, bip_size), dtype=float)
                for r in range(bip_size):
                    for c in range(bip_size):
                        bips[r, c] = float(self.bip_table.item(r, c).text()) if self.bip_table.item(r, c) else 0.0

                if EOSModelParameters.__name__ != "EOSModelParameters":
                    # Fallback if core data models failed to import
                    output['eos_model'] = {
                        'eos_type': self.eos_combo.currentText(),
                        'component_properties': comp_props,
                        'binary_interaction_coeffs': bips
                    }
                else:
                    output['eos_model'] = EOSModelParameters(
                        eos_type=self.eos_combo.currentText(),
                        component_properties=comp_props,
                        binary_interaction_coeffs=bips
                    )
        except (ValueError, TypeError) as e:
            logger.error(f"Could not parse EOS data: {e}"); output['eos_model'] = None
            
        return output

    def load_data(self, data: Dict[str, Any]):
        """Populates the dialog widgets from a data dictionary."""
        self.ref_pressure_edit.setText(str(data.get('ref_pressure', '')))
        self.ref_temp_edit.setText(str(data.get('ref_temperature', '')))
        for name, editor in self.pvt_editors.items():
            if name in data and isinstance(data[name], np.ndarray):
                editor.set_full_data(data[name])

        # --- INTEGRATION: Load data into the EOS tab ---
        eos_model = data.get('eos_model')
        if eos_model:
            # Handle both dict and dataclass formats for robustness
            eos_params = eos_model if isinstance(eos_model, EOSModelParameters) else EOSModelParameters(**eos_model)
            
            self.eos_combo.setCurrentText(eos_params.eos_type)
            
            comp_props = eos_params.component_properties
            self.comp_table.setRowCount(comp_props.shape[0])
            for r in range(comp_props.shape[0]):
                for c in range(comp_props.shape[1]):
                    self.comp_table.setItem(r, c, QTableWidgetItem(str(comp_props[r, c])))
            
            # Bip table must be updated based on loaded components
            self._update_bip_table()
            bips = eos_params.binary_interaction_coeffs
            for r in range(min(bips.shape[0], self.bip_table.rowCount())):
                for c in range(min(bips.shape[1], self.bip_table.columnCount())):
                    self.bip_table.setItem(r, c, QTableWidgetItem(str(bips[r, c])))