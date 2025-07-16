import logging
from typing import Optional, Any, Dict, List

import numpy as np

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QGridLayout, QGroupBox, QLabel,
    QComboBox, QPushButton, QMessageBox, QDoubleSpinBox, QHBoxLayout,
    QSplitter, QFormLayout, QTableWidget, QAbstractItemView, QTableWidgetItem
)
from PyQt6.QtGui import QIcon
from PyQt6.QtCore import Qt, pyqtSignal

import plotly.graph_objects as go
from PyQt6.QtWebEngineWidgets import QWebEngineView

try:
    from .workers.well_analysis_worker import WellAnalysisWorker
    from co2eor_optimizer.core.data_models import WellData, PVTProperties
    from co2eor_optimizer.evaluation.mmp import MMP_METHODS
    _import_failed = False
except ImportError as e:
    logging.critical(f"MMPWellAnalysisWidget: Failed to import critical components: {e}")
    class WellData:
        def __init__(self, name: str = "Dummy Well", metadata: dict = {}):
            self.name = name
            self.metadata = metadata
    class PVTProperties: ...
    class WellAnalysisWorker:
        result_ready = pyqtSignal(dict)
        error_occurred = pyqtSignal(str)
        finished = pyqtSignal()
        def __init__(self, *args, **kwargs): ...
        def start(self): pass
        def isRunning(self): return False
        def stop(self): pass
        def wait(self, timeout: int): pass
        def deleteLater(self): pass
    MMP_METHODS = {"dummy_method": lambda x: x}
    _import_failed = True

logger = logging.getLogger(__name__)

class MMPWellAnalysisWidget(QWidget):
    analysis_completed = pyqtSignal(dict)
    # [NEW] Signal to emit the single, representative MMP value for the engine
    representative_mmp_calculated = pyqtSignal(float)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        
        self.well_data_list: List[WellData] = []
        self.pvt_data: Optional[PVTProperties] = None
        self.worker: Optional[WellAnalysisWorker] = None

        self._setup_ui()
        self._connect_signals()
        
        if _import_failed:
            self.setEnabled(False)
            QMessageBox.critical(self, "Component Error", "Core components for well analysis are missing. This tab will be disabled.")
        else:
            self.update_data([], None) 

    def _setup_ui(self):
        """Sets up the main UI with a splitter for a more dynamic layout."""
        main_layout = QHBoxLayout(self)
        
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        config_panel = self._create_config_panel()
        results_panel = self._create_results_panel()
        
        splitter.addWidget(config_panel)
        splitter.addWidget(results_panel)
        splitter.setSizes([350, 650])
        
        main_layout.addWidget(splitter)

    def _create_config_panel(self) -> QWidget:
        """Creates the left-side panel with all user controls."""
        container = QWidget()
        layout = QVBoxLayout(container)

        config_group = QGroupBox("Analysis Configuration")
        form_layout = QFormLayout(config_group)
        self.well_select_combo = QComboBox()
        form_layout.addRow("Select Well:", self.well_select_combo)
        self.mmp_method_combo = QComboBox()
        self.mmp_method_combo.addItem("Auto-Select", "auto")
        for name, func in MMP_METHODS.items():
            self.mmp_method_combo.addItem(name.replace("_", " ").title(), name)
        form_layout.addRow("MMP Method:", self.mmp_method_combo)
        layout.addWidget(config_group)

        self.method_inputs_group = QGroupBox("Method-Specific Inputs")
        method_inputs_layout = QVBoxLayout(self.method_inputs_group)
        
        self.c7_mw_widget = QWidget()
        c7_layout = QHBoxLayout(self.c7_mw_widget); c7_layout.setContentsMargins(0,0,0,0)
        c7_layout.addWidget(QLabel("C7+ Molecular Weight (g/mol):"))
        self.c7_mw_input = QDoubleSpinBox(); self.c7_mw_input.setRange(50.0, 250.0); self.c7_mw_input.setValue(190.0)
        self.c7_mw_input.setToolTip("Used for the 'Hybrid GH' and 'Alston' correlations.")
        c7_layout.addWidget(self.c7_mw_input)
        method_inputs_layout.addWidget(self.c7_mw_widget)

        self.gas_comp_widget = QWidget()
        gas_layout = QGridLayout(self.gas_comp_widget); gas_layout.setContentsMargins(0,0,0,0)
        gas_layout.addWidget(QLabel("<b>Gas Composition (Mole Fraction)</b>"), 0, 0, 1, 2)
        gas_layout.addWidget(QLabel("CO₂:"), 1, 0)
        self.co2_comp_input = QDoubleSpinBox(); self.co2_comp_input.setRange(0.0, 1.0); self.co2_comp_input.setValue(1.0); self.co2_comp_input.setDecimals(3)
        gas_layout.addWidget(self.co2_comp_input, 1, 1)
        gas_layout.addWidget(QLabel("CH₄ (Methane):"), 2, 0)
        self.ch4_comp_input = QDoubleSpinBox(); self.ch4_comp_input.setRange(0.0, 1.0); self.ch4_comp_input.setValue(0.0); self.ch4_comp_input.setDecimals(3)
        gas_layout.addWidget(self.ch4_comp_input, 2, 1)

        gas_layout.addWidget(QLabel("N₂ (Nitrogen):"), 3, 0)
        self.n2_comp_input = QDoubleSpinBox()
        self.n2_comp_input.setRange(0.0, 1.0); self.n2_comp_input.setValue(0.0); self.n2_comp_input.setDecimals(3)
        self.n2_comp_input.setToolTip("Used for the 'Alston' correlation.")
        gas_layout.addWidget(self.n2_comp_input, 3, 1)
        
        normalize_btn = QPushButton("Normalize to 1.0"); normalize_btn.clicked.connect(self._normalize_gas_composition)
        gas_layout.addWidget(normalize_btn, 4, 0, 1, 2)

        method_inputs_layout.addWidget(self.gas_comp_widget)
        
        self.method_inputs_group.setVisible(False)
        layout.addWidget(self.method_inputs_group)

        layout.addStretch()
        
        self.calculate_button = QPushButton(QIcon.fromTheme("system-run"), " Calculate MMP Profile")
        layout.addWidget(self.calculate_button)

        return container

    def _create_results_panel(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        results_group = QGroupBox("Analysis Results")
        results_layout = QVBoxLayout(results_group)

        v_splitter = QSplitter(Qt.Orientation.Vertical)

        self.plot_view = QWebEngineView()
        self.plot_view.setMinimumHeight(300)
        v_splitter.addWidget(self.plot_view)

        self.results_table = QTableWidget()
        self.results_table.setColumnCount(4)
        self.results_table.setHorizontalHeaderLabels(["Depth (ft)", "Temperature (°F)", "Oil Gravity (°API)", "MMP (psia)"])
        self.results_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.results_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.results_table.setAlternatingRowColors(True)
        self.results_table.setMinimumHeight(150)
        v_splitter.addWidget(self.results_table)

        v_splitter.setSizes([400, 200])

        results_layout.addWidget(v_splitter, 1)

        self.status_label = QLabel("<i>Load project data to perform analysis.</i>")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        results_layout.addWidget(self.status_label)
        layout.addWidget(results_group)
        return container

    def _connect_signals(self):
        self.calculate_button.clicked.connect(self._run_mmp_calculation)
        self.well_select_combo.currentIndexChanged.connect(self._update_method_availability)
        self.mmp_method_combo.currentTextChanged.connect(self._update_method_specific_inputs_visibility)

    def _normalize_gas_composition(self):
        co2 = self.co2_comp_input.value()
        ch4 = self.ch4_comp_input.value()
        n2 = self.n2_comp_input.value()
        total = co2 + ch4 + n2
        if total > 1e-6:
            self.co2_comp_input.setValue(co2 / total)
            self.ch4_comp_input.setValue(ch4 / total)
            self.n2_comp_input.setValue(n2 / total)

    def _update_method_specific_inputs_visibility(self):
        selected_method = self.mmp_method_combo.currentData()
        show_c7_mw = selected_method in ['hybrid_gh', 'alston']
        show_gas_comp = selected_method in ['yuan', 'alston']
        
        self.c7_mw_widget.setVisible(show_c7_mw)
        self.gas_comp_widget.setVisible(show_gas_comp)
        self.method_inputs_group.setVisible(show_c7_mw or show_gas_comp)

    def _update_method_availability(self):
        current_well = self.well_select_combo.currentData()
        if not isinstance(current_well, WellData):
            for i in range(self.mmp_method_combo.count()):
                if self.mmp_method_combo.itemData(i) not in ['auto', 'cronquist']: self.mmp_method_combo.model().item(i).setEnabled(False)
            return

        for i in range(self.mmp_method_combo.count()): self.mmp_method_combo.model().item(i).setEnabled(True)
        self._update_method_specific_inputs_visibility()

    def update_data(self, wells: List[WellData], pvt: Optional[PVTProperties]):
        logger.info(f"MMPWidget updating with {len(wells)} wells."); self.well_data_list, self.pvt_data = wells, pvt
        self.well_select_combo.blockSignals(True); self.well_select_combo.clear()
        if not self.well_data_list: self.well_select_combo.addItem("No wells loaded"); self.well_select_combo.setEnabled(False)
        else:
            for well in self.well_data_list: self.well_select_combo.addItem(well.name, userData=well)
            self.well_select_combo.setEnabled(True)
        self.well_select_combo.blockSignals(False); self._update_button_state(); self._update_method_availability()

    def _update_button_state(self):
        can_calculate = bool(self.well_data_list and self.pvt_data)
        self.calculate_button.setEnabled(can_calculate)
        self.status_label.setText("<i>Ready to calculate MMP profile.</i>" if can_calculate else "<i>Well and/or PVT data is missing.</i>")

    def _run_mmp_calculation(self):
        if self.worker and self.worker.isRunning(): QMessageBox.warning(self, "Busy", "An analysis is already in progress."); return
        well, method = self.well_select_combo.currentData(), self.mmp_method_combo.currentData()
        if not all([well, self.pvt_data, method]): QMessageBox.warning(self, "Missing Data", "A valid well, PVT data, and calculation method are required."); return

        worker_kwargs = {'method': method}
        if self.c7_mw_widget.isVisible(): worker_kwargs['c7_plus_mw_override'] = self.c7_mw_input.value()
        if self.gas_comp_widget.isVisible():
            self._normalize_gas_composition()
            gas_comp = {
                'CO2': self.co2_comp_input.value(),
                'CH4': self.ch4_comp_input.value(),
                'N2': self.n2_comp_input.value()
            }
            worker_kwargs['gas_composition'] = {k: v for k, v in gas_comp.items() if v > 1e-6}

        self.status_label.setText(f"<i>Calculating MMP profile for {well.name}...</i>"); self.calculate_button.setEnabled(False)
        self.plot_view.setHtml("<p style='text-align:center;'><i>Processing...</i></p>")
        self.worker = WellAnalysisWorker(well_data=well, pvt_data=self.pvt_data, **worker_kwargs)
        self.worker.result_ready.connect(self._on_mmp_result)
        self.worker.error_occurred.connect(self._on_mmp_error)
        self.worker.finished.connect(self._on_worker_finished)
        self.worker.start()

    def _populate_results_table(self, results: Dict[str, np.ndarray]):
        """Populates the table with a concise summary of results, showing only distinct property zones."""
        self.results_table.setRowCount(0)
        depths = results.get('depths', np.array([]))
        mmp = results.get('mmp', np.array([]))
        temp = results.get('temperature', np.array([]))
        api = results.get('api', np.array([]))
        
        if depths.size == 0:
            return

        valid_mask = ~np.isnan(mmp)
        if not np.any(valid_mask):
            return

        valid_depths = depths[valid_mask]
        valid_temp = temp[valid_mask]
        valid_api = api[valid_mask]
        valid_mmp = mmp[valid_mask]

        unique_rows_to_display = []
        if valid_depths.size > 0:
            first_row = (valid_depths[0], valid_temp[0], valid_api[0], valid_mmp[0])
            unique_rows_to_display.append(first_row)
            
            last_added_props = (first_row[1], first_row[2], first_row[3])

            for i in range(1, len(valid_depths)):
                current_props = (valid_temp[i], valid_api[i], valid_mmp[i])
                
                if not np.allclose(current_props, last_added_props, equal_nan=True):
                    row_to_add = (valid_depths[i], valid_temp[i], valid_api[i], valid_mmp[i])
                    unique_rows_to_display.append(row_to_add)
                    last_added_props = current_props

        self.results_table.setRowCount(len(unique_rows_to_display))
        for row_idx, row_data in enumerate(unique_rows_to_display):
            d, t, a, m = row_data
            self.results_table.setItem(row_idx, 0, QTableWidgetItem(f"{d:.1f}"))
            self.results_table.setItem(row_idx, 1, QTableWidgetItem(f"{t:.2f}"))
            self.results_table.setItem(row_idx, 2, QTableWidgetItem(f"{a:.2f}"))
            self.results_table.setItem(row_idx, 3, QTableWidgetItem(f"{m:.2f}"))
        
        self.results_table.resizeColumnsToContents()

    def _on_mmp_result(self, results: Dict[str, list]):
        well_name = self.well_select_combo.currentText()
        self.status_label.setText(f"MMP profile calculation complete for {well_name}.")
        try:
            processed_results = {key: np.array(value) for key, value in results.items() if isinstance(value, list)}
            
            num_points = processed_results.get('depths', np.array([])).size
            if num_points > 0:
                self.plot_view.setHtml(self._create_mmp_plot(processed_results, well_name).to_html(include_plotlyjs='cdn'))
                self._populate_results_table(processed_results)
                
                # [MODIFIED] Calculate and emit the representative MMP value
                mmp_values = processed_results.get('mmp', np.array([]))
                if mmp_values.size > 0 and np.any(~np.isnan(mmp_values)):
                    representative_mmp = np.nanmean(mmp_values)
                    logger.info(f"Emitting representative MMP: {representative_mmp:.2f} psi")
                    self.representative_mmp_calculated.emit(representative_mmp)
            else:
                self._on_mmp_error("Analysis returned no data points.")
            
            self.analysis_completed.emit(results)
        except Exception as e:
            logger.error(f"Error processing MMP results: {e}", exc_info=True)
            self._on_mmp_error(f"Failed to process results: {e}")

    def _create_mmp_plot(self, results: Dict[str, np.ndarray], well_name: str) -> go.Figure:
        """Creates a multi-axis plot showing MMP and Temperature vs. Depth."""
        depths, mmp, temp, api = (results.get('depths'), results.get('mmp'), 
                                  results.get('temperature'), results.get('api'))

        if depths is None or depths.size == 0 or mmp is None or mmp.size == 0:
            raise ValueError("Result dictionary is missing 'depths' or 'mmp' data.")

        fig = go.Figure()

        temp_for_plot = np.copy(temp)
        temp_for_plot[np.isnan(mmp)] = np.nan
        fig.add_trace(go.Scatter(
            x=temp_for_plot, y=depths, name='Temperature', mode='lines',
            line=dict(color='crimson', dash='dash', width=2),
            xaxis='x2',
            connectgaps=False
        ))

        customdata = np.stack((temp, api), axis=-1)
        hovertemplate = ('<b>Depth: %{y:.1f} ft</b><br>' +
                         'MMP: %{x:.2f} psia<br>' +
                         'Temperature: %{customdata[0]:.2f} °F<br>' +
                         'API Gravity: %{customdata[1]:.2f}°<extra></extra>')
        
        fig.add_trace(go.Scatter(
            x=mmp, y=depths, name='MMP', mode='lines',
            line=dict(color='royalblue', width=4, shape='hv'),
            customdata=customdata,
            hovertemplate=hovertemplate,
            connectgaps=False
        ))

        fig.update_layout(
            title=f"MMP & Temperature Profile for Well: <b>{well_name}</b>",
            xaxis_title="MMP (psia)",
            yaxis_title="Depth (ft)",
            yaxis=dict(autorange="reversed"),
            xaxis2=dict(
                title="Temperature (°F)",
                overlaying="x",
                side="top",
                showgrid=False
            ),
            legend=dict(yanchor="top", y=0.98, xanchor="left", x=0.02,
                        bgcolor='rgba(255,255,255,0.7)'),
            margin=dict(t=80)
        )
        return fig

    def _on_mmp_error(self, error_message: str):
        self.status_label.setText(f"<p style='color:red;'><b>Error:</b> {error_message}</p>")
        self.plot_view.setHtml(f"<p style='text-align:center; color:red;'>Calculation failed.<br>{error_message}</p>")
        logger.error(f"MMP calculation error: {error_message}")

    def _on_worker_finished(self):
        self._update_button_state()
        if self.worker:
            self.worker.deleteLater()
            self.worker = None

    def closeEvent(self, event: Any):
        if self.worker and self.worker.isRunning():
            logger.info("Stopping ongoing WellAnalysisWorker due to widget close.")
            self.worker.stop()
            self.worker.wait(1000)
        super().closeEvent(event)