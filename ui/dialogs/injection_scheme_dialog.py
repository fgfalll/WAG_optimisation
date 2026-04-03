import logging
from typing import Optional, Dict, Any
from dataclasses import asdict

from PyQt6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QWidget,
    QTabWidget,
    QGroupBox,
    QFormLayout,
    QComboBox,
    QDoubleSpinBox,
    QSpinBox,
    QCheckBox,
    QPushButton,
    QLabel,
    QFrame,
    QScrollArea,
    QTextEdit,
    QListWidget,
    QListWidgetItem,
    QStackedWidget,
    QSplitter,
)
from PyQt6.QtGui import QIcon, QPixmap, QPainter, QColor
from PyQt6.QtCore import Qt, pyqtSignal, QSize

from core.data_models import EORParameters

logger = logging.getLogger(__name__)


class SchemePreviewWidget(QWidget):
    """Widget for visualizing injection scheme timelines."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setMinimumHeight(150)
        self.setMaximumHeight(200)
        self.scheme_data: Dict[str, Any] = {}

    def set_scheme_data(self, scheme_type: str, parameters: Dict[str, Any]):
        """Set the scheme data for visualization."""
        self.scheme_data = {"type": scheme_type, "parameters": parameters}
        self.update()

    def paintEvent(self, event):
        """Paint the scheme timeline visualization."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Clear background
        painter.fillRect(self.rect(), QColor(240, 240, 240))

        if not self.scheme_data:
            painter.drawText(
                self.rect(), Qt.AlignmentFlag.AlignCenter, "Select a scheme to preview"
            )
            return

        scheme_type = self.scheme_data["type"]
        params = self.scheme_data["parameters"]

        # Draw timeline based on scheme type
        width = self.width() - 40
        height = self.height() - 40
        x_offset = 20
        y_offset = 20

        painter.setPen(QColor(0, 0, 0))
        painter.drawRect(x_offset, y_offset, width, height)

        if scheme_type == "continuous":
            self._draw_continuous_scheme(painter, x_offset, y_offset, width, height)
        elif scheme_type == "wag":
            self._draw_wag_scheme(painter, x_offset, y_offset, width, height, params)
        elif scheme_type == "huff_n_puff":
            self._draw_huff_n_puff_scheme(painter, x_offset, y_offset, width, height, params)
        elif scheme_type == "swag":
            self._draw_swag_scheme(painter, x_offset, y_offset, width, height, params)
        elif scheme_type == "tapered":
            self._draw_tapered_scheme(painter, x_offset, y_offset, width, height, params)
        elif scheme_type == "pulsed":
            self._draw_pulsed_scheme(painter, x_offset, y_offset, width, height, params)

    def _draw_continuous_scheme(self, painter, x, y, width, height):
        """Draw continuous injection scheme."""
        painter.setBrush(QColor(0, 100, 200))  # Blue for CO2
        painter.drawRect(x, y + height // 2, width, height // 4)
        painter.drawText(x, y - 5, "Continuous CO2 Injection")

    def _draw_wag_scheme(self, painter, x, y, width, height, params):
        """Draw WAG scheme."""
        wag_ratio = params.get("WAG_ratio", 1.0)
        min_cycle = params.get("min_cycle_length_days", 30)
        max_cycle = params.get("max_cycle_length_days", 180)

        # Calculate phase widths based on WAG ratio
        total_ratio = wag_ratio + 1.0
        water_fraction = wag_ratio / total_ratio
        co2_fraction = 1.0 / total_ratio

        cycles = min(6, width // 30)  # Limit cycles for display
        avg_cycle = (min_cycle + max_cycle) // 2

        painter.setPen(QColor(0, 0, 0))
        painter.drawText(x, y - 5, f"WAG Ratio: {wag_ratio:.1f}, Avg Cycle: {avg_cycle}d")

        for i in range(cycles):
            cycle_x = x + i * (width // cycles)
            cycle_width = width // cycles

            # CO2 phase (blue) - based on ratio
            co2_width = int(cycle_width * co2_fraction)
            painter.setBrush(QColor(0, 100, 200))
            painter.drawRect(cycle_x, y + height // 2, co2_width, height // 4)

            # Water phase (cyan) - based on ratio
            water_width = int(cycle_width * water_fraction)
            painter.setBrush(QColor(0, 200, 200))
            painter.drawRect(cycle_x + co2_width, y + height // 2, water_width, height // 4)

            # Draw cycle separator
            if i < cycles - 1:
                painter.setPen(QColor(100, 100, 100))
                painter.drawLine(
                    cycle_x + cycle_width,
                    y + height // 2,
                    cycle_x + cycle_width,
                    y + height // 2 + height // 4,
                )
                painter.setPen(QColor(0, 0, 0))

    def _draw_huff_n_puff_scheme(self, painter, x, y, width, height, params):
        """Draw Huff-n-Puff scheme."""
        injection_days = params.get("huff_n_puff_injection_period_days", 30)
        soak_days = params.get("huff_n_puff_soaking_period_days", 15)
        prod_days = params.get("huff_n_puff_production_period_days", 45)
        total_cycle = injection_days + soak_days + prod_days
        cycles = min(3, width // 30)

        painter.setPen(QColor(0, 0, 0))
        painter.drawText(
            x, y - 5, f"Huff-n-Puff: {injection_days}d inj, {soak_days}d soak, {prod_days}d prod"
        )

        for i in range(cycles):
            cycle_x = x + i * (width // cycles)
            cycle_width = width // cycles

            # Injection phase (blue)
            inj_width = int(cycle_width * injection_days / total_cycle)
            painter.setBrush(QColor(0, 100, 200))
            painter.drawRect(cycle_x, y + height // 2, inj_width, height // 4)

            # Soak phase (gray)
            soak_x = cycle_x + inj_width
            soak_width = int(cycle_width * soak_days / total_cycle)
            painter.setBrush(QColor(150, 150, 150))
            painter.drawRect(soak_x, y + height // 2, soak_width, height // 4)

            # Production phase (green)
            prod_x = soak_x + soak_width
            prod_width = int(cycle_width * prod_days / total_cycle)
            painter.setBrush(QColor(0, 200, 0))
            painter.drawRect(prod_x, y + height // 2, prod_width, height // 4)

            # Draw cycle separator
            if i < cycles - 1:
                painter.setPen(QColor(100, 100, 100))
                painter.drawLine(
                    cycle_x + cycle_width,
                    y + height // 2,
                    cycle_x + cycle_width,
                    y + height // 2 + height // 4,
                )
                painter.setPen(QColor(0, 0, 0))

    def _draw_swag_scheme(self, painter, x, y, width, height, params):
        """Draw SWAG scheme."""
        water_gas_ratio = params.get("swag_water_gas_ratio", 1.0)
        simultaneous = params.get("swag_simultaneous_injection", True)

        painter.setPen(QColor(0, 0, 0))
        mode = "Simultaneous" if simultaneous else "Alternating"
        painter.drawText(x, y - 5, f"SWAG: {mode}, WGR: {water_gas_ratio:.1f}")

        if simultaneous:
            # Draw simultaneous injection (overlapping bars)
            painter.setBrush(QColor(0, 100, 200, 180))  # CO2 with transparency
            painter.drawRect(x, y + height // 3, width, height // 6)

            painter.setBrush(QColor(0, 200, 200, 180))  # Water with transparency
            painter.drawRect(x, y + height // 2, width, height // 6)
        else:
            # Draw alternating injection
            co2_width = int(width * 0.6)
            water_width = int(width * 0.4)

            painter.setBrush(QColor(0, 100, 200))
            painter.drawRect(x, y + height // 2, co2_width, height // 4)

            painter.setBrush(QColor(0, 200, 200))
            painter.drawRect(x + co2_width, y + height // 2, water_width, height // 4)

    def _draw_tapered_scheme(self, painter, x, y, width, height, params):
        """Draw tapered injection scheme."""
        initial_rate = params.get("tapered_initial_rate_multiplier", 2.0)
        final_rate = params.get("tapered_final_rate_multiplier", 0.5)
        duration = params.get("tapered_duration_years", 5.0)
        function = params.get("tapered_function", "linear")

        painter.setPen(QColor(0, 0, 0))
        painter.drawText(
            x, y - 5, f"Tapered: {initial_rate:.1f}→{final_rate:.1f} over {duration}yr ({function})"
        )

        # Draw tapered line based on function
        painter.setPen(QColor(0, 100, 200))
        start_y = y + height // 4
        end_y = y + 3 * height // 4

        if function == "linear":
            painter.drawLine(x, start_y, x + width, end_y)
        elif function == "exponential":
            # Draw exponential curve
            points = []
            for i in range(10):
                px = x + i * width // 9
                t = i / 9.0
                py = start_y + (end_y - start_y) * (1 - t**2)
                points.append((px, int(py)))
            for i in range(len(points) - 1):
                painter.drawLine(points[i][0], points[i][1], points[i + 1][0], points[i + 1][1])
        else:  # logarithmic
            # Draw logarithmic curve
            points = []
            for i in range(10):
                px = x + i * width // 9
                t = i / 9.0
                py = start_y + (end_y - start_y) * (1 - (1 - t) ** 2)
                points.append((px, int(py)))
            for i in range(len(points) - 1):
                painter.drawLine(points[i][0], points[i][1], points[i + 1][0], points[i + 1][1])

    def _draw_pulsed_scheme(self, painter, x, y, width, height, params):
        """Draw pulsed injection scheme."""
        pulse_days = params.get("pulsed_pulse_duration_days", 7)
        pause_days = params.get("pulsed_pause_duration_days", 14)
        intensity = params.get("pulsed_intensity_multiplier", 2.0)
        total_cycle = pulse_days + pause_days
        pulses = min(8, width // 15)

        painter.setPen(QColor(0, 0, 0))
        painter.drawText(
            x, y - 5, f"Pulsed: {pulse_days}d on, {pause_days}d off, {intensity:.1f}x intensity"
        )

        for i in range(pulses):
            pulse_x = x + i * (width // pulses)
            pulse_width = int(width // pulses * pulse_days / total_cycle)

            # Draw pulse with intensity-based height
            pulse_height = int(height // 4 * intensity / 2.0)  # Scale height by intensity
            pulse_y = y + height // 2 - pulse_height // 4

            painter.setBrush(QColor(0, 100, 200))
            painter.drawRect(pulse_x, pulse_y, pulse_width, pulse_height)

            # Draw pause period
            if i < pulses - 1:
                pause_x = pulse_x + pulse_width
                pause_width = int(width // pulses * pause_days / total_cycle)
                painter.setPen(QColor(100, 100, 100))
                painter.drawLine(pause_x, y + height // 2, pause_x + pause_width, y + height // 2)
                painter.setPen(QColor(0, 0, 0))


class InjectionSchemeDialog(QDialog):
    """Dialog for managing sophisticated injection schemes."""

    scheme_updated = pyqtSignal(dict)  # Emits updated EOR parameters

    def __init__(self, eor_parameters: EORParameters, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.eor_parameters = eor_parameters
        self.original_parameters = asdict(eor_parameters)
        self.current_scheme = eor_parameters.injection_scheme

        self.setWindowTitle("Injection Scheme Configuration")
        self.setMinimumSize(800, 600)
        self._setup_ui()
        self._select_current_scheme()

    def _setup_ui(self):
        """Setup the dialog UI."""
        main_layout = QVBoxLayout(self)

        # Scheme selection area
        scheme_selection_layout = QHBoxLayout()

        # Scheme list
        self.scheme_list_widget = QListWidget()
        self.scheme_list_widget.setMaximumWidth(200)
        schemes = [
            ("Continuous", "Constant CO2 injection", "continuous"),
            ("WAG", "Water-Alternating-Gas injection", "wag"),
            ("Huff-n-Puff", "Cyclic injection with soak periods", "huff_n_puff"),
            ("SWAG", "Simultaneous Water and Gas", "swag"),
            ("Tapered", "Gradually decreasing injection", "tapered"),
            ("Pulsed", "Intermittent high-intensity injection", "pulsed"),
        ]

        for name, description, scheme_type in schemes:
            item = QListWidgetItem(f"{name}\n{description}")
            item.setData(Qt.ItemDataRole.UserRole, scheme_type)
            self.scheme_list_widget.addItem(item)

        self.scheme_list_widget.currentItemChanged.connect(self._on_scheme_selected)
        scheme_selection_layout.addWidget(self.scheme_list_widget)

        # Scheme parameters area
        self.scheme_stack = QStackedWidget()
        scheme_selection_layout.addWidget(self.scheme_stack)

        # Create parameter widgets for each scheme
        self._create_scheme_parameter_widgets()

        main_layout.addLayout(scheme_selection_layout)

        # Preview area
        preview_group = QGroupBox("Scheme Preview")
        preview_layout = QVBoxLayout(preview_group)
        self.preview_widget = SchemePreviewWidget()
        preview_layout.addWidget(self.preview_widget)
        main_layout.addWidget(preview_group)

        # Buttons
        button_layout = QHBoxLayout()
        self.apply_btn = QPushButton("Apply Scheme")
        self.apply_btn.clicked.connect(self._apply_scheme)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.reject)
        self.reset_btn = QPushButton("Reset to Defaults")
        self.reset_btn.clicked.connect(self._reset_to_defaults)

        button_layout.addWidget(self.reset_btn)
        button_layout.addStretch()
        button_layout.addWidget(self.cancel_btn)
        button_layout.addWidget(self.apply_btn)
        main_layout.addLayout(button_layout)

    def _create_scheme_parameter_widgets(self):
        """Create parameter configuration widgets for each scheme."""
        # Continuous scheme (minimal parameters)
        continuous_widget = QWidget()
        continuous_layout = QFormLayout(continuous_widget)
        continuous_layout.addRow(QLabel("Continuous injection requires no additional parameters."))
        self.scheme_stack.addWidget(continuous_widget)

        # WAG scheme
        wag_widget = QWidget()
        wag_layout = QFormLayout(wag_widget)
        self.wag_ratio_spin = QDoubleSpinBox()
        self.wag_ratio_spin.setRange(0.1, 5.0)
        self.wag_ratio_spin.setValue(self.eor_parameters.WAG_ratio or 1.0)
        self.wag_ratio_spin.valueChanged.connect(self._update_preview)
        wag_layout.addRow("WAG Ratio:", self.wag_ratio_spin)

        self.min_cycle_spin = QSpinBox()
        self.min_cycle_spin.setRange(1, 365)
        self.min_cycle_spin.setValue(int(self.eor_parameters.min_cycle_length_days))
        self.min_cycle_spin.valueChanged.connect(self._update_preview)
        wag_layout.addRow("Min Cycle Length (days):", self.min_cycle_spin)

        self.max_cycle_spin = QSpinBox()
        self.max_cycle_spin.setRange(1, 365)
        self.max_cycle_spin.setValue(int(self.eor_parameters.max_cycle_length_days))
        self.max_cycle_spin.valueChanged.connect(self._update_preview)
        wag_layout.addRow("Max Cycle Length (days):", self.max_cycle_spin)
        self.scheme_stack.addWidget(wag_widget)

        # Huff-n-Puff scheme
        hnp_widget = QWidget()
        hnp_layout = QFormLayout(hnp_widget)

        self.hnp_cycle_spin = QSpinBox()
        self.hnp_cycle_spin.setRange(30, 365)
        self.hnp_cycle_spin.setValue(int(self.eor_parameters.huff_n_puff_cycle_length_days))
        self.hnp_cycle_spin.valueChanged.connect(self._update_preview)
        hnp_layout.addRow("Cycle Length (days):", self.hnp_cycle_spin)

        self.hnp_inject_spin = QSpinBox()
        self.hnp_inject_spin.setRange(1, 90)
        self.hnp_inject_spin.setValue(int(self.eor_parameters.huff_n_puff_injection_period_days))
        self.hnp_inject_spin.valueChanged.connect(self._update_preview)
        hnp_layout.addRow("Injection Period (days):", self.hnp_inject_spin)

        self.hnp_soak_spin = QSpinBox()
        self.hnp_soak_spin.setRange(0, 60)
        self.hnp_soak_spin.setValue(int(self.eor_parameters.huff_n_puff_soaking_period_days))
        self.hnp_soak_spin.valueChanged.connect(self._update_preview)
        hnp_layout.addRow("Soaking Period (days):", self.hnp_soak_spin)

        self.hnp_prod_spin = QSpinBox()
        self.hnp_prod_spin.setRange(1, 180)
        self.hnp_prod_spin.setValue(int(self.eor_parameters.huff_n_puff_production_period_days))
        self.hnp_prod_spin.valueChanged.connect(self._update_preview)
        hnp_layout.addRow("Production Period (days):", self.hnp_prod_spin)

        self.hnp_max_cycles_spin = QSpinBox()
        self.hnp_max_cycles_spin.setRange(1, 50)
        self.hnp_max_cycles_spin.setValue(int(self.eor_parameters.huff_n_puff_max_cycles))
        self.hnp_max_cycles_spin.valueChanged.connect(self._update_preview)
        hnp_layout.addRow("Maximum Cycles:", self.hnp_max_cycles_spin)
        self.scheme_stack.addWidget(hnp_widget)

        # SWAG scheme
        swag_widget = QWidget()
        swag_layout = QFormLayout(swag_widget)

        self.swag_ratio_spin = QDoubleSpinBox()
        self.swag_ratio_spin.setRange(0.1, 5.0)
        self.swag_ratio_spin.setValue(self.eor_parameters.swag_water_gas_ratio)
        self.swag_ratio_spin.valueChanged.connect(self._update_preview)
        swag_layout.addRow("Water-Gas Ratio:", self.swag_ratio_spin)

        self.swag_simultaneous_cb = QCheckBox()
        self.swag_simultaneous_cb.setChecked(self.eor_parameters.swag_simultaneous_injection)
        self.swag_simultaneous_cb.stateChanged.connect(self._update_preview)
        swag_layout.addRow("Simultaneous Injection:", self.swag_simultaneous_cb)

        self.swag_efficiency_spin = QDoubleSpinBox()
        self.swag_efficiency_spin.setRange(0.0, 1.0)
        self.swag_efficiency_spin.setSingleStep(0.1)
        self.swag_efficiency_spin.setValue(self.eor_parameters.swag_mixing_efficiency)
        self.swag_efficiency_spin.valueChanged.connect(self._update_preview)
        swag_layout.addRow("Mixing Efficiency:", self.swag_efficiency_spin)
        self.scheme_stack.addWidget(swag_widget)

        # Tapered scheme
        tapered_widget = QWidget()
        tapered_layout = QFormLayout(tapered_widget)

        self.tapered_initial_spin = QDoubleSpinBox()
        self.tapered_initial_spin.setRange(0.5, 3.0)
        self.tapered_initial_spin.setValue(self.eor_parameters.tapered_initial_rate_multiplier)
        self.tapered_initial_spin.valueChanged.connect(self._update_preview)
        tapered_layout.addRow("Initial Rate Multiplier:", self.tapered_initial_spin)

        self.tapered_final_spin = QDoubleSpinBox()
        self.tapered_final_spin.setRange(0.1, 1.0)
        self.tapered_final_spin.setValue(self.eor_parameters.tapered_final_rate_multiplier)
        self.tapered_final_spin.valueChanged.connect(self._update_preview)
        tapered_layout.addRow("Final Rate Multiplier:", self.tapered_final_spin)

        self.tapered_duration_spin = QDoubleSpinBox()
        self.tapered_duration_spin.setRange(1.0, 20.0)
        self.tapered_duration_spin.setValue(self.eor_parameters.tapered_duration_years)
        self.tapered_duration_spin.valueChanged.connect(self._update_preview)
        tapered_layout.addRow("Tapering Duration (years):", self.tapered_duration_spin)

        self.tapered_function_combo = QComboBox()
        self.tapered_function_combo.addItems(["linear", "exponential", "logarithmic"])
        self.tapered_function_combo.setCurrentText(self.eor_parameters.tapered_function)
        self.tapered_function_combo.currentTextChanged.connect(self._update_preview)
        tapered_layout.addRow("Tapering Function:", self.tapered_function_combo)
        self.scheme_stack.addWidget(tapered_widget)

        # Pulsed scheme
        pulsed_widget = QWidget()
        pulsed_layout = QFormLayout(pulsed_widget)

        self.pulsed_pulse_spin = QSpinBox()
        self.pulsed_pulse_spin.setRange(1, 30)
        self.pulsed_pulse_spin.setValue(int(self.eor_parameters.pulsed_pulse_duration_days))
        self.pulsed_pulse_spin.valueChanged.connect(self._update_preview)
        pulsed_layout.addRow("Pulse Duration (days):", self.pulsed_pulse_spin)

        self.pulsed_pause_spin = QSpinBox()
        self.pulsed_pause_spin.setRange(1, 60)
        self.pulsed_pause_spin.setValue(int(self.eor_parameters.pulsed_pause_duration_days))
        self.pulsed_pause_spin.valueChanged.connect(self._update_preview)
        pulsed_layout.addRow("Pause Duration (days):", self.pulsed_pause_spin)

        self.pulsed_intensity_spin = QDoubleSpinBox()
        self.pulsed_intensity_spin.setRange(1.0, 5.0)
        self.pulsed_intensity_spin.setValue(self.eor_parameters.pulsed_intensity_multiplier)
        self.pulsed_intensity_spin.valueChanged.connect(self._update_preview)
        pulsed_layout.addRow("Intensity Multiplier:", self.pulsed_intensity_spin)
        self.scheme_stack.addWidget(pulsed_widget)

    def _select_current_scheme(self):
        """Select the current scheme in the list widget based on the EOR parameters."""
        for i in range(self.scheme_list_widget.count()):
            item = self.scheme_list_widget.item(i)
            scheme_type = item.data(Qt.ItemDataRole.UserRole)
            if scheme_type == self.current_scheme:
                self.scheme_list_widget.setCurrentItem(item)
                break

        # Update preview with current parameters
        self._update_preview()

    def _on_scheme_selected(self, current, previous):
        """Handle scheme selection change."""
        if current is None:
            return

        scheme_type = current.data(Qt.ItemDataRole.UserRole)
        self.current_scheme = scheme_type

        # Show corresponding parameter widget
        scheme_index_map = {
            "continuous": 0,
            "wag": 1,
            "huff_n_puff": 2,
            "swag": 3,
            "tapered": 4,
            "pulsed": 5,
        }

        if scheme_type in scheme_index_map:
            self.scheme_stack.setCurrentIndex(scheme_index_map[scheme_type])

        # Update preview
        self._update_preview()

    def _update_preview(self):
        """Update the scheme preview widget."""
        params = self._get_current_parameters()
        self.preview_widget.set_scheme_data(self.current_scheme, params)

    def _get_current_parameters(self) -> Dict[str, Any]:
        """Get current parameter values for the selected scheme."""
        params = {}

        if self.current_scheme == "wag":
            params.update(
                {
                    "WAG_ratio": self.wag_ratio_spin.value(),
                    "min_cycle_length_days": self.min_cycle_spin.value(),
                    "max_cycle_length_days": self.max_cycle_spin.value(),
                }
            )
        elif self.current_scheme == "huff_n_puff":
            params.update(
                {
                    "huff_n_puff_cycle_length_days": self.hnp_cycle_spin.value(),
                    "huff_n_puff_injection_period_days": self.hnp_inject_spin.value(),
                    "huff_n_puff_soaking_period_days": self.hnp_soak_spin.value(),
                    "huff_n_puff_production_period_days": self.hnp_prod_spin.value(),
                    "huff_n_puff_max_cycles": self.hnp_max_cycles_spin.value(),
                }
            )
        elif self.current_scheme == "swag":
            params.update(
                {
                    "swag_water_gas_ratio": self.swag_ratio_spin.value(),
                    "swag_simultaneous_injection": self.swag_simultaneous_cb.isChecked(),
                    "swag_mixing_efficiency": self.swag_efficiency_spin.value(),
                }
            )
        elif self.current_scheme == "tapered":
            params.update(
                {
                    "tapered_initial_rate_multiplier": self.tapered_initial_spin.value(),
                    "tapered_final_rate_multiplier": self.tapered_final_spin.value(),
                    "tapered_duration_years": self.tapered_duration_spin.value(),
                    "tapered_function": self.tapered_function_combo.currentText(),
                }
            )
        elif self.current_scheme == "pulsed":
            params.update(
                {
                    "pulsed_pulse_duration_days": self.pulsed_pulse_spin.value(),
                    "pulsed_pause_duration_days": self.pulsed_pause_spin.value(),
                    "pulsed_intensity_multiplier": self.pulsed_intensity_spin.value(),
                }
            )

        return params

    def _apply_scheme(self):
        """Apply the selected scheme configuration."""
        try:
            # Update EOR parameters with current values
            updated_params = self.original_parameters.copy()
            updated_params["injection_scheme"] = self.current_scheme

            # Update scheme-specific parameters
            current_params = self._get_current_parameters()
            updated_params.update(current_params)

            # Validate the parameters
            temp_eor = EORParameters(**updated_params)

            # Emit the updated parameters
            self.scheme_updated.emit(updated_params)
            self.accept()

        except Exception as e:
            logger.error(f"Error applying injection scheme: {e}")
            # Show error message to user
            from PyQt6.QtWidgets import QMessageBox

            QMessageBox.critical(self, "Validation Error", f"Invalid scheme parameters: {e}")

    def _reset_to_defaults(self):
        """Reset scheme parameters to defaults."""
        default_eor = EORParameters()
        default_params = asdict(default_eor)

        # Update UI with default values
        self.wag_ratio_spin.setValue(default_params.get("WAG_ratio", 1.0))
        self.min_cycle_spin.setValue(default_params.get("min_cycle_length_days", 30))
        self.max_cycle_spin.setValue(default_params.get("max_cycle_length_days", 180))

        # Update other scheme parameters similarly...

        self._update_preview()

    def validate_for_engine(self, engine_type: str) -> bool:
        """Validate injection scheme parameters for specific engine type.

        This method performs engine-specific validation of injection scheme parameters
        to ensure compatibility with the selected simulation engine.

        Args:
            engine_type: The engine type ("simple", "detailed", or "surrogate")

        Returns:
            True if parameters are valid for the engine, False otherwise
        """
        try:
            # Get current parameters
            current_params = self._get_current_parameters()

            if engine_type == "simple":
                # Simple engine only needs basic params
                required_params = ["injection_rate", "target_pressure_psi", "max_pressure_psi"]
                for param in required_params:
                    value = self.original_parameters.get(param)
                    if value is not None and value <= 0:
                        logger.warning(f"Simple engine validation: {param} must be positive")
                        return False

                # WAG scheme validation for simple engine
                if self.current_scheme == "wag":
                    wag_ratio = current_params.get("WAG_ratio", 1.0)
                    if not (0.1 <= wag_ratio <= 5.0):
                        logger.warning(f"Simple engine: WAG ratio {wag_ratio} out of range [0.1, 5.0]")
                        return False

            elif engine_type == "detailed":
                # Detailed engine needs all parameters with stricter validation
                required_params = [
                    "injection_rate", "target_pressure_psi", "max_pressure_psi",
                    "mobility_ratio", "injection_scheme"
                ]
                for param in required_params:
                    value = self.original_parameters.get(param)
                    if value is None or value <= 0:
                        logger.warning(f"Detailed engine validation: {param} is required and must be positive")
                        return False

                # Pressure constraints for detailed engine
                target_p = self.original_parameters.get("target_pressure_psi", 0)
                max_p = self.original_parameters.get("max_pressure_psi", 0)
                if max_p < target_p:
                    logger.warning(f"Detailed engine: Max pressure {max_p} must be >= target pressure {target_p}")
                    return False

                # Temperature validation for detailed engine
                temp = self.original_parameters.get("temperature", 0)
                if not (50 <= temp <= 300):
                    logger.warning(f"Detailed engine: Temperature {temp} must be in range [50, 300]°F")
                    return False

                # Mobility ratio validation for detailed engine
                mobility = self.original_parameters.get("mobility_ratio", 0)
                if not (1.0 <= mobility <= 50.0):
                    logger.warning(f"Detailed engine: Mobility ratio {mobility} must be in range [1.0, 50.0]")
                    return False

            elif engine_type == "surrogate":
                # Surrogate has minimal requirements - just basic validation
                injection_rate = self.original_parameters.get("injection_rate", 0)
                if injection_rate <= 0:
                    logger.warning("Surrogate engine: injection_rate must be positive")
                    return False

            logger.info(f"Injection scheme validation passed for {engine_type} engine")
            return True

        except Exception as e:
            logger.error(f"Error during engine-specific validation: {e}")
            return False
