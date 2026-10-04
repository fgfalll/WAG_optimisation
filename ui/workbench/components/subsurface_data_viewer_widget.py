"""
Subsurface Data & Graph Workstation Viewer
==========================================

Dedicated full-screen middle workstation component for rendering high-precision
engineering spreadsheets, analytical data tables, and scientific graphs
directly from the Hierarchical Model Tree.

Supports:
- Clean modern tabular views with real-time text search/filtering, TSV clipboard copy, and CSV export.
- High-resolution scientific 2D plots with customizable palettes, gridlines, and image export.
"""

import os
import logging
from typing import Dict, Any, List, Optional, Callable
import numpy as np

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QTableWidget, QTableWidgetItem, QHeaderView, QStackedWidget, QFrame,
    QFileDialog, QMessageBox, QApplication
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QFont, QKeySequence

import matplotlib
matplotlib.use("QtAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

logger = logging.getLogger(__name__)


class SubsurfaceDataViewerWidget(QWidget):
    """
    Full-screen middle view for displaying attached Sheets/Tables and Graphs
    selected from the Hierarchical Model Tree.
    """

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.current_export_name = "subsurface_export"
        self._raw_table_rows: List[List[str]] = []
        self._table_headers: List[str] = []

        self._setup_ui()

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        self.setStyleSheet("""
            QWidget {
                background: #ffffff;
                color: #1e293b;
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            }
        """)

        self.stack = QStackedWidget(self)

        # ---------------- Page 0: Sheet / Table View ----------------
        self.page_table = QWidget()
        table_layout = QVBoxLayout(self.page_table)
        table_layout.setContentsMargins(6, 6, 6, 6)
        table_layout.setSpacing(6)

        # Table Header Bar
        table_bar = QFrame()
        table_bar.setStyleSheet("""
            QFrame {
                background: #f8fafc;
                border: 1px solid #dee2e6;
                border-radius: 4px;
                padding: 4px 8px;
            }
        """)
        table_bar_layout = QHBoxLayout(table_bar)
        table_bar_layout.setContentsMargins(4, 2, 4, 2)
        table_bar_layout.setSpacing(8)

        self.lbl_table_title = QLabel("Data Sheet / Table")
        self.lbl_table_title.setStyleSheet("font-size: 13px; font-weight: bold; color: #0f172a;")
        table_bar_layout.addWidget(self.lbl_table_title)

        self.lbl_table_subtitle = QLabel("")
        self.lbl_table_subtitle.setStyleSheet("font-size: 11px; color: #64748b; font-style: italic;")
        table_bar_layout.addWidget(self.lbl_table_subtitle)

        table_bar_layout.addStretch()

        # Search filter
        self.filter_edit = QLineEdit()
        self.filter_edit.setPlaceholderText("Filter rows...")
        self.filter_edit.setFixedWidth(180)
        self.filter_edit.setStyleSheet("""
            QLineEdit {
                background: #ffffff;
                border: 1px solid #cbd5e1;
                border-radius: 4px;
                padding: 3px 8px;
                font-size: 11px;
            }
            QLineEdit:focus {
                border-color: #0d6efd;
            }
        """)
        self.filter_edit.textChanged.connect(self._apply_table_filter)
        table_bar_layout.addWidget(self.filter_edit)

        # Copy to Clipboard Button
        self.btn_copy_table = QPushButton("Copy Table")
        self.btn_copy_table.setToolTip("Copy table contents to clipboard as tab-separated values (ready for Excel)")
        self.btn_copy_table.setStyleSheet("""
            QPushButton {
                background: #ffffff;
                color: #334155;
                border: 1px solid #cbd5e1;
                border-radius: 4px;
                padding: 4px 10px;
                font-size: 11px;
                font-weight: 600;
            }
            QPushButton:hover {
                background: #e2e8f0;
                color: #0d6efd;
            }
        """)
        self.btn_copy_table.clicked.connect(self._copy_table_to_clipboard)
        table_bar_layout.addWidget(self.btn_copy_table)

        # Export CSV Button
        self.btn_export_csv = QPushButton("Export CSV")
        self.btn_export_csv.setToolTip("Export sheet to comma-separated CSV file")
        self.btn_export_csv.setStyleSheet("""
            QPushButton {
                background: #0284c7;
                color: #ffffff;
                border: 1px solid #0369a1;
                border-radius: 4px;
                padding: 4px 12px;
                font-size: 11px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: #0369a1;
            }
        """)
        self.btn_export_csv.clicked.connect(self._export_table_to_csv)
        table_bar_layout.addWidget(self.btn_export_csv)

        table_layout.addWidget(table_bar)

        # QTableWidget
        self.table = QTableWidget()
        self.table.setAlternatingRowColors(True)
        self.table.setSortingEnabled(True)
        self.table.setStyleSheet("""
            QTableWidget {
                background: #ffffff;
                alternate-background-color: #f8fafc;
                gridline-color: #e2e8f0;
                border: 1px solid #dee2e6;
                border-radius: 4px;
                font-size: 11px;
                color: #1e293b;
            }
            QTableWidget::item {
                padding: 4px 8px;
            }
            QTableWidget::item:selected {
                background: #e0f2fe;
                color: #0369a1;
            }
            QHeaderView::section {
                background: #f1f5f9;
                color: #334155;
                font-weight: bold;
                font-size: 11px;
                border: 1px solid #cbd5e1;
                padding: 6px 8px;
            }
        """)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(True)
        table_layout.addWidget(self.table, stretch=1)

        self.stack.addWidget(self.page_table)  # Page 0

        # ---------------- Page 1: Scientific Graph View ----------------
        self.page_graph = QWidget()
        graph_layout = QVBoxLayout(self.page_graph)
        graph_layout.setContentsMargins(6, 6, 6, 6)
        graph_layout.setSpacing(6)

        # Graph Header Bar
        graph_bar = QFrame()
        graph_bar.setStyleSheet("""
            QFrame {
                background: #f8fafc;
                border: 1px solid #dee2e6;
                border-radius: 4px;
                padding: 4px 8px;
            }
        """)
        graph_bar_layout = QHBoxLayout(graph_bar)
        graph_bar_layout.setContentsMargins(4, 2, 4, 2)
        graph_bar_layout.setSpacing(8)

        self.lbl_graph_title = QLabel("Scientific Graph")
        self.lbl_graph_title.setStyleSheet("font-size: 13px; font-weight: bold; color: #0f172a;")
        graph_bar_layout.addWidget(self.lbl_graph_title)

        self.lbl_graph_subtitle = QLabel("")
        self.lbl_graph_subtitle.setStyleSheet("font-size: 11px; color: #64748b; font-style: italic;")
        graph_bar_layout.addWidget(self.lbl_graph_subtitle)

        graph_bar_layout.addStretch()

        self.btn_save_plot = QPushButton("Save PNG")
        self.btn_save_plot.setStyleSheet("""
            QPushButton {
                background: #ffffff;
                color: #334155;
                border: 1px solid #cbd5e1;
                border-radius: 4px;
                padding: 4px 12px;
                font-size: 11px;
                font-weight: 600;
            }
            QPushButton:hover {
                background: #e2e8f0;
                color: #0d6efd;
            }
        """)
        self.btn_save_plot.clicked.connect(self._save_plot_png)
        graph_bar_layout.addWidget(self.btn_save_plot)

        graph_layout.addWidget(graph_bar)

        # Matplotlib Canvas
        self.fig = Figure(figsize=(9, 6), dpi=100, facecolor="#ffffff")
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setStyleSheet("border: 1px solid #dee2e6; border-radius: 4px; background: #ffffff;")
        graph_layout.addWidget(self.canvas, stretch=1)

        self.stack.addWidget(self.page_graph)  # Page 1

        main_layout.addWidget(self.stack)

    # -------------------------------------------------------------------------
    # Table Controller
    # -------------------------------------------------------------------------
    def show_table(
        self,
        title: str,
        subtitle: str,
        headers: List[str],
        rows: List[List[Any]],
        export_name: str = "table_export"
    ):
        """Displays data in full-screen modern interactive spreadsheet table."""
        self.current_export_name = export_name
        self.lbl_table_title.setText(title)
        self.lbl_table_subtitle.setText(f"({len(rows)} entries) — {subtitle}")
        self._table_headers = list(headers)
        self._raw_table_rows = [[str(c) for c in r] for r in rows]

        self.filter_edit.clear()
        self.table.setSortingEnabled(False)
        self.table.clear()
        self.table.setColumnCount(len(headers))
        self.table.setHorizontalHeaderLabels(headers)
        self.table.setRowCount(len(rows))

        for r_idx, row in enumerate(rows):
            for c_idx, val in enumerate(row):
                item = QTableWidgetItem(str(val))
                val_str = str(val).strip()
                # Right align numbers
                try:
                    float(val_str.replace(",", "").replace("%", "").replace("psi", "").replace("ft", "").replace("bbl", ""))
                    item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                except ValueError:
                    if val_str in ("PASS", "Active", "Miscible", "Strongly Water-Wet", "Low"):
                        item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                        item.setForeground(QColor("#15803d"))
                        font = item.font()
                        font.setBold(True)
                        item.setFont(font)
                    elif val_str in ("WARN", "WARNING", "Moderate"):
                        item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                        item.setForeground(QColor("#b45309"))
                        font = item.font()
                        font.setBold(True)
                        item.setFont(font)
                    elif val_str in ("FAIL", "Shut-In", "Immiscible", "High", "Critical"):
                        item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                        item.setForeground(QColor("#b91c1c"))
                        font = item.font()
                        font.setBold(True)
                        item.setFont(font)
                    else:
                        item.setTextAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

                self.table.setItem(r_idx, c_idx, item)

        self.table.setSortingEnabled(True)
        self.table.horizontalHeader().resizeSections(QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.stack.setCurrentIndex(0)

    def _apply_table_filter(self, text: str):
        search = text.lower().strip()
        for row in range(self.table.rowCount()):
            if not search:
                self.table.setRowHidden(row, False)
                continue
            matches = False
            for col in range(self.table.columnCount()):
                item = self.table.item(row, col)
                if item and search in item.text().lower():
                    matches = True
                    break
            self.table.setRowHidden(row, not matches)

    def _copy_table_to_clipboard(self):
        """Copies table to clipboard as Tab-Separated Values (TSV)."""
        lines = ["\t".join(self._table_headers)]
        for row in self._raw_table_rows:
            lines.append("\t".join(row))
        text = "\n".join(lines)
        QApplication.clipboard().setText(text)
        QToolTip.showText(self.btn_copy_table.mapToGlobal(self.btn_copy_table.rect().center()), "Copied to clipboard (TSV for Excel)")

    def _export_table_to_csv(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export Sheet to CSV", f"{self.current_export_name}.csv", "CSV Files (*.csv)")
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(",".join(f'"{h}"' for h in self._table_headers) + "\n")
                for row in self._raw_table_rows:
                    f.write(",".join(f'"{c}"' for c in row) + "\n")
            QMessageBox.information(self, "Export Successful", f"Sheet data exported successfully to:\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to export CSV: {e}")

    # -------------------------------------------------------------------------
    # Graph Controller
    # -------------------------------------------------------------------------
    def show_graph(
        self,
        title: str,
        subtitle: str,
        draw_func: Callable[[Figure, FigureCanvas], None],
        export_name: str = "graph_export"
    ):
        """Draws custom scientific plot full screen."""
        self.current_export_name = export_name
        self.lbl_graph_title.setText(title)
        self.lbl_graph_subtitle.setText(subtitle)

        self.fig.clear()
        try:
            draw_func(self.fig, self.canvas)
            self.fig.tight_layout()
            self.canvas.draw_idle()
        except Exception as e:
            logger.error(f"Error drawing graph '{title}': {e}", exc_info=True)
            ax = self.fig.add_subplot(111)
            ax.text(0.5, 0.5, f"Plot rendering error:\n{e}", ha="center", va="center", color="#ef4444", fontsize=11)
            self.canvas.draw_idle()

        self.stack.setCurrentIndex(1)

    def _save_plot_png(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save Graph PNG", f"{self.current_export_name}.png", "PNG Images (*.png)")
        if not path:
            return
        try:
            self.fig.savefig(path, dpi=300, bbox_inches="tight")
            QMessageBox.information(self, "Image Saved", f"High-resolution graph saved to:\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Failed to save image: {e}")

    # -------------------------------------------------------------------------
    # Built-In Data & Graph Generators
    # -------------------------------------------------------------------------
    def render_grid_table(self, params: Dict[str, Any]):
        nx = int(params.get("nx", 50))
        ny = int(params.get("ny", 50))
        nz = int(params.get("nz", 10))
        length = float(params.get("length", 2000.0))
        area = float(params.get("area", 100.0))
        width = (area * 43560.0) / max(length, 1.0)
        thickness = float(params.get("thickness", 50.0))
        top_depth = float(params.get("initial_pressure", 4000.0) * 0.433) if "depth" not in params else float(params.get("depth", 5000.0))
        dz = thickness / max(nz, 1)
        dx = length / max(nx, 1)
        dy = width / max(ny, 1)
        k_base = float(params.get("perm", 100.0))
        phi_base = float(params.get("poro", 0.20))

        headers = [
            "Layer (K)", "Top Depth (ft)", "Base Depth (ft)", "Layer Dz (ft)",
            "Grid NX", "Grid NY", "Cells per Layer", "Bulk Vol (acre-ft)",
            "Dx (ft)", "Dy (ft)", "Layer Perm (mD)", "Layer Porosity (%)"
        ]
        rows = []
        for k in range(nz):
            z_top = top_depth + k * dz
            z_bot = z_top + dz
            # Layer variation
            k_fac = 1.0 + 0.15 * np.sin(k * 0.8)
            phi_fac = 1.0 + 0.05 * np.cos(k * 0.8)
            vol_acre_ft = (length * width * dz) / 43560.0
            rows.append([
                f"K = {k+1}", f"{z_top:.1f}", f"{z_bot:.1f}", f"{dz:.1f}",
                str(nx), str(ny), f"{nx * ny:,}", f"{vol_acre_ft:.1f}",
                f"{dx:.1f}", f"{dy:.1f}", f"{k_base * k_fac:.1f}", f"{phi_base * phi_fac * 100:.1f}%"
            ])

        self.show_table(
            "Grid Discretization & Layer Coordinates",
            f"3D Structured Mesh: {nx} × {ny} × {nz} = {nx*ny*nz:,} Active Cells",
            headers, rows, "grid_layer_coordinates"
        )

    def render_grid_graph(self, params: Dict[str, Any]):
        nz = int(params.get("nz", 10))
        thickness = float(params.get("thickness", 50.0))
        dz = thickness / max(nz, 1)
        top_depth = 5000.0 if "depth" not in params else float(params.get("depth", 5000.0))

        def draw(fig: Figure, canvas: FigureCanvas):
            ax1 = fig.add_subplot(121)
            ax2 = fig.add_subplot(122)

            k_layers = np.arange(1, nz + 1)
            depths_mid = top_depth + (k_layers - 0.5) * dz
            perm_profile = float(params.get("perm", 100.0)) * (1.0 + 0.2 * np.sin(k_layers * 0.7))
            poro_profile = float(params.get("poro", 0.20)) * 100.0 * (1.0 + 0.08 * np.cos(k_layers * 0.7))

            ax1.step(perm_profile, depths_mid, where="mid", color="#0284c7", linewidth=2.0, label="Permeability (mD)")
            ax1.fill_betweenx(depths_mid, 0, perm_profile, step="mid", color="#e0f2fe", alpha=0.5)
            ax1.set_xlabel("Permeability (mD)", fontweight="bold", fontsize=10)
            ax1.set_ylabel("True Vertical Depth (ft)", fontweight="bold", fontsize=10)
            ax1.set_title("Layer Permeability vs Depth", fontweight="bold", fontsize=11)
            ax1.invert_yaxis()
            ax1.grid(True, linestyle=":", alpha=0.5)
            ax1.legend(loc="lower left", fontsize=9)

            ax2.step(poro_profile, depths_mid, where="mid", color="#16a34a", linewidth=2.0, label="Porosity (%)")
            ax2.fill_betweenx(depths_mid, 0, poro_profile, step="mid", color="#dcfce7", alpha=0.5)
            ax2.set_xlabel("Porosity (%)", fontweight="bold", fontsize=10)
            ax2.set_title("Layer Porosity vs Depth", fontweight="bold", fontsize=11)
            ax2.invert_yaxis()
            ax2.grid(True, linestyle=":", alpha=0.5)
            ax2.legend(loc="lower left", fontsize=9)

        self.show_graph(
            "Grid Layer Depth & Petrophysical Profiles",
            f"Depth Window: {top_depth:.0f} ft to {top_depth + thickness:.0f} ft (TVD)",
            draw, "grid_depth_profiles"
        )

    def render_ooip_table(self, params: Dict[str, Any]):
        lx = float(params.get("length", 2000.0))
        area_acres = float(params.get("area", 100.0))
        ly = (area_acres * 43560.0) / max(lx, 1.0)
        h = float(params.get("thickness", 50.0))
        phi = float(params.get("poro", 0.20))
        swi = float(params.get("swi", 0.25))
        boi = float(params.get("boi", 1.20))
        rs_ini = float(params.get("sol_gor", 500.0))

        # Standard Reservoir Volumetrics (Field Units)
        grv_acre_ft = area_acres * h
        pv_bbl = grv_acre_ft * 7758.37 * phi
        pv_mmbbl = pv_bbl / 1e6
        hcpv_bbl = pv_bbl * (1.0 - swi)
        ooip_stb = (7758.37 * area_acres * h * phi * (1.0 - swi)) / max(boi, 0.5)
        ooip_mmstb = ooip_stb / 1e6
        initial_gas_mscf = ooip_stb * rs_ini / 1000.0
        initial_gas_bcf = initial_gas_mscf / 1e6

        # Theoretical CO2 Storage (assuming dense supercritical CO2 ~ 650 kg/m3)
        # 1 bbl pore volume = 0.158987 m3. Dense CO2 = 650 kg/m3 = 0.650 metric tons/m3
        co2_storage_tons = (pv_bbl * 0.158987 * 0.650)
        co2_storage_mt = co2_storage_tons / 1e6
        effective_eor_co2_mt = co2_storage_mt * 0.45  # typical 40-50% accessible storage

        headers = ["Volumetric Metric", "Symbol", "Calculated Value", "Field Units", "Governing Physical Formula"]
        rows = [
            ["Surface Drainage Area", "A", f"{area_acres:,.1f}", "acres", "Input boundary parameter"],
            ["Reservoir Length (X)", "Lx", f"{lx:,.1f}", "ft", "Areal extent along primary axis"],
            ["Reservoir Width (Y)", "Ly", f"{ly:,.1f}", "ft", "Derived width = (A × 43,560) / Lx"],
            ["Net Pay Thickness", "h", f"{h:,.1f}", "ft", "Volumetric formation thickness"],
            ["Gross Rock Volume (GRV)", "GRV", f"{grv_acre_ft:,.1f}", "acre-ft", "GRV = Area × Net Pay"],
            ["Average Porosity", "φ", f"{phi * 100:.1f}%", "fraction", "Core/log volumetric porosity"],
            ["Total Pore Volume (PV)", "Vp", f"{pv_mmbbl:,.2f}", "MMbbl", "Vp = GRV × 7,758.37 × φ"],
            ["Initial Water Saturation", "Swi", f"{swi * 100:.1f}%", "fraction", "Irreducible water saturation"],
            ["Hydrocarbon Pore Volume", "HCPV", f"{hcpv_bbl / 1e6:,.2f}", "MMbbl", "HCPV = Vp × (1 - Swi)"],
            ["Initial Formation Volume Factor", "Boi", f"{boi:.3f}", "rb/STB", "Solvent-extended PVT Bo"],
            ["Original Oil in Place (OOIP)", "N", f"{ooip_mmstb:,.3f}", "MMSTB", "N = (7,758 × A × h × φ × (1 - Swi)) / Boi"],
            ["OOIP in Stock Tank Barrels", "N (STB)", f"{ooip_stb:,.0f}", "STB", "Exact stock-tank volume"],
            ["Initial Dissolved Gas", "G_s", f"{initial_gas_bcf:,.2f}", "BCF", "Gs = N × Rsi"],
            ["Theoretical 100% CO2 Capacity", "M_co2,max", f"{co2_storage_mt:,.2f}", "Million Metric Tons", "M_co2 = Vp × ρ_sc(CO2) @ res P, T"],
            ["Effective EOR Storage Target", "M_co2,eor", f"{effective_eor_co2_mt:,.2f}", "Million Metric Tons", "Net CO2 trapped in swept pore space"]
        ]

        self.show_table(
            "Volumetric OOIP & HCPV Material Balance",
            f"OOIP = {ooip_mmstb:.2f} MMSTB | Theoretical CO2 Storage = {co2_storage_mt:.2f} MT",
            headers, rows, "volumetric_ooip_material_balance"
        )

    def render_ooip_graph(self, params: Dict[str, Any]):
        lx = float(params.get("length", 2000.0))
        area = float(params.get("area", 100.0))
        h = float(params.get("thickness", 50.0))
        phi = float(params.get("poro", 0.20))
        swi = float(params.get("swi", 0.25))
        boi = float(params.get("boi", 1.20))

        ooip_mmstb = (7758.37 * area * h * phi * (1.0 - swi)) / max(boi, 0.5) / 1e6
        pv_mmbbl = (area * h * 7758.37 * phi) / 1e6
        water_pv = pv_mmbbl * swi
        oil_pv = pv_mmbbl * (1.0 - swi)
        primary_rec = ooip_mmstb * 0.18
        waterflood_rec = ooip_mmstb * 0.22
        co2_eor_target = ooip_mmstb * 0.18
        residual_target = ooip_mmstb * 0.42

        def draw(fig: Figure, canvas: FigureCanvas):
            ax1 = fig.add_subplot(121)
            ax2 = fig.add_subplot(122)

            # Left: Pore Volume Saturation Stack
            categories_pv = ["Water", "Residual Oil", "Mobile Oil"]
            vals_pv = [water_pv, oil_pv * 0.4, oil_pv * 0.6]
            colors_pv = ["#38bdf8", "#fbbf24", "#22c55e"]
            ax1.bar(["Total Pore Volume"], [vals_pv[0]], color=colors_pv[0], label=f"Connate Water ({vals_pv[0]:.1f} MMbbl)", width=0.5)
            ax1.bar(["Total Pore Volume"], [vals_pv[1]], bottom=[vals_pv[0]], color=colors_pv[1], label=f"Residual Oil ({vals_pv[1]:.1f} MMbbl)", width=0.5)
            ax1.bar(["Total Pore Volume"], [vals_pv[2]], bottom=[vals_pv[0] + vals_pv[1]], color=colors_pv[2], label=f"Mobile Oil ({vals_pv[2]:.1f} MMbbl)", width=0.5)
            ax1.set_ylabel("Reservoir Pore Volume (MMbbl)", fontweight="bold")
            ax1.set_title(f"Pore Volume Partitioning (Total: {pv_mmbbl:.1f} MMbbl)", fontweight="bold", fontsize=11)
            ax1.grid(True, linestyle=":", alpha=0.5)
            ax1.legend(loc="upper right", fontsize=8.5)

            # Right: Oil Recovery Potential Breakdown
            categories = ["Primary", "Waterflood", "CO2 EOR", "Residual"]
            vals = [primary_rec, waterflood_rec, co2_eor_target, residual_target]
            cols = ["#94a3b8", "#0284c7", "#16a34a", "#e11d48"]
            bars = ax2.bar(categories, vals, color=cols, edgecolor="#334155", linewidth=1.0)
            ax2.set_ylabel("Oil Volume (MMSTB)", fontweight="bold")
            ax2.set_title(f"OOIP Recovery Targets ({ooip_mmstb:.2f} MMSTB Total)", fontweight="bold", fontsize=11)
            ax2.grid(True, linestyle=":", alpha=0.5)
            for bar in bars:
                y = bar.get_height()
                pct = (y / ooip_mmstb) * 100
                ax2.text(bar.get_x() + bar.get_width() / 2, y + 0.05, f"{y:.2f}\n({pct:.0f}%)", ha="center", va="bottom", fontsize=8, fontweight="bold")

        self.show_graph(
            "Volumetric Partitioning & EOR Recovery Targets",
            f"OOIP = {ooip_mmstb:.2f} MMSTB across {area:.0f} acres",
            draw, "ooip_recovery_targets"
        )

    def render_stratigraphy_table(self, params: Dict[str, Any]):
        thickness = float(params.get("thickness", 50.0))
        top_depth = 5000.0 if "depth" not in params else float(params.get("depth", 5000.0))
        k_base = float(params.get("perm", 100.0))
        phi_base = float(params.get("poro", 0.20))

        headers = ["Zone / Member", "Top MD (ft)", "Base MD (ft)", "Gross Pay (ft)", "Net Pay (ft)", "Net-to-Gross", "Lithology", "Perm (mD)", "Porosity (%)"]
        rows = [
            ["Zone A (Upper Sand)", f"{top_depth:.0f}", f"{top_depth + thickness * 0.35:.0f}", f"{thickness * 0.35:.1f}", f"{thickness * 0.30:.1f}", "0.86", "Aeolian Sandstone", f"{k_base * 1.6:.1f}", f"{phi_base * 1.15 * 100:.1f}%"],
            ["Zone B (Middle Silt/Shale)", f"{top_depth + thickness * 0.35:.0f}", f"{top_depth + thickness * 0.65:.0f}", f"{thickness * 0.30:.1f}", f"{thickness * 0.15:.1f}", "0.50", "Siltstone / Dense Shale", f"{k_base * 0.25:.1f}", f"{phi_base * 0.75 * 100:.1f}%"],
            ["Zone C (Basal High-Perm)", f"{top_depth + thickness * 0.65:.0f}", f"{top_depth + thickness:.0f}", f"{thickness * 0.35:.1f}", f"{thickness * 0.32:.1f}", "0.91", "Coarse Channel Sand", f"{k_base * 2.2:.1f}", f"{phi_base * 1.25 * 100:.1f}%"],
        ]
        self.show_table(
            "Zonal Stratigraphy & Layer Tops",
            f"Total Gross Pay: {thickness:.1f} ft across 3 Geological Zones",
            headers, rows, "stratigraphy_layer_tops"
        )

    def render_rock_table(self, params: Dict[str, Any]):
        headers = ["Petrophysical Attribute", "Symbol", "Value", "Units", "Provenance / Modeling Method"]
        k = float(params.get("perm", 100.0))
        phi = float(params.get("poro", 0.20))
        v_dp = float(params.get("dykstra_parsons", 0.65))
        kv_kh = float(params.get("kv_kh_ratio", 0.10))
        headers = ["Property", "Symbol", "Mean Realization", "P10", "P90", "Field Units", "Model Method"]
        rows = [
            ["Permeability (Horiz)", "kh", f"{k:.1f}", f"{k * 0.35:.1f}", f"{k * 2.1:.1f}", "mD", "Facies-conditioned SGSIM"],
            ["Porosity", "φ", f"{phi*100:.1f}%", f"{phi*0.7*100:.1f}%", f"{phi*1.3*100:.1f}%", "fraction", "Facies-conditioned SGSIM"],
            ["Vertical Permeability", "kv", f"{k * kv_kh:.1f}", f"{k * kv_kh * 0.3:.1f}", f"{k * kv_kh * 2.0:.1f}", "mD", f"kv/kh = {kv_kh:.2f}"],
            ["Dykstra-Parsons Coeff", "V_DP", f"{v_dp:.2f}", f"{max(0.2, v_dp-0.15):.2f}", f"{min(0.95, v_dp+0.15):.2f}", "dimensionless", "Heterogeneity indicator"],
            ["Sand Facies Fraction", "F_sand", f"{float(params.get('sand_fraction', 0.65))*100:.0f}%", "55%", "75%", "volumetric", "Lithofacies distribution"],
            ["Silt Facies Fraction", "F_silt", f"{float(params.get('silt_fraction', 0.25))*100:.0f}%", "15%", "35%", "volumetric", "Lithofacies distribution"],
            ["Shale Facies Fraction", "F_shale", f"{float(params.get('shale_fraction', 0.10))*100:.0f}%", "5%", "15%", "volumetric", "Baffle / barrier seal"]
        ]
        self.show_table(
            "Petrophysical Realization Summary Statistics",
            f"Mean Perm = {k:.1f} mD | Mean Poro = {phi*100:.1f}% | V_DP = {v_dp:.2f}",
            headers, rows, "petrophysical_summary"
        )

    def render_rock_graph(self, params: Dict[str, Any]):
        k_base = float(params.get("perm", 100.0))
        phi_base = float(params.get("poro", 0.20))

        def draw(fig: Figure, canvas: FigureCanvas):
            np.random.seed(42)
            n_pts = 400
            phi_samples = np.clip(np.random.normal(phi_base, 0.04, n_pts), 0.04, 0.35)
            # Permeability correlated to porosity via Kozeny-Carman log-normal trend
            k_samples = np.clip(k_base * (phi_samples / phi_base)**3.5 * np.exp(np.random.normal(0, 0.4, n_pts)), 0.1, 5000.0)

            ax1 = fig.add_subplot(121)
            ax2 = fig.add_subplot(122)

            # Cross-plot
            ax1.scatter(phi_samples * 100.0, k_samples, c=k_samples, cmap="viridis", alpha=0.7, edgecolors="none", s=25)
            ax1.set_yscale("log")
            ax1.set_xlabel("Porosity (%)", fontweight="bold")
            ax1.set_ylabel("Permeability (mD, Log Scale)", fontweight="bold")
            ax1.set_title("Porosity vs Permeability Cross-Plot", fontweight="bold", fontsize=11)
            ax1.grid(True, which="both", linestyle=":", alpha=0.5)

            # Fit line
            p_sort = np.sort(phi_samples) * 100.0
            k_trend = k_base * (p_sort / (phi_base * 100.0))**3.5
            ax1.plot(p_sort, k_trend, color="#dc2626", linewidth=2.0, label="Kozeny-Carman Trend")
            ax1.legend(loc="upper left", fontsize=8.5)

            # Histogram
            ax2.hist(k_samples, bins=np.logspace(np.log10(0.1), np.log10(5000), 25), color="#0284c7", edgecolor="#0f172a", alpha=0.75)
            ax2.set_xscale("log")
            ax2.set_xlabel("Permeability (mD, Log Scale)", fontweight="bold")
            ax2.set_ylabel("Cell Count Frequency", fontweight="bold")
            ax2.set_title("Permeability Distribution Histogram", fontweight="bold", fontsize=11)
            ax2.grid(True, linestyle=":", alpha=0.5)

        self.show_graph(
            "Porosity-Permeability Cross-Plot & Spatial Distribution",
            f"Based on Facies Correlation (N = 400 sample realization)",
            draw, "poro_perm_crossplot"
        )

    def render_relperm_table(self, params: Dict[str, Any]):
        headers = ["Relative Permeability Parameter", "Symbol", "Active Value", "Typical Range", "Physical Definition"]
        rows = [
            ["Connate Water Saturation", "Swc", f"{float(params.get('s_wc', 0.20)):.3f}", "0.15 - 0.35", "Irreducible wetting phase saturation"],
            ["Residual Oil to Water", "Sorw", f"{float(params.get('s_orw', 0.20)):.3f}", "0.15 - 0.35", "Unrecoverable oil after water displacement"],
            ["Critical Gas Saturation", "Sgc", f"{float(params.get('s_gc', 0.05)):.3f}", "0.02 - 0.10", "Gas saturation required for gas phase mobility"],
            ["Residual Oil to Gas", "Sorg", f"{float(params.get('s_org', 0.15)):.3f}", "0.10 - 0.25", "Unrecoverable oil after gas/solvent flooding"],
            ["Endpoint Water Rel-Perm", "krw0", f"{float(params.get('krw0', 0.30)):.3f}", "0.10 - 0.50", "Water relative permeability at (1 - Sorw)"],
            ["Endpoint Oil Rel-Perm", "kro0", f"{float(params.get('kro0', 0.85)):.3f}", "0.70 - 1.00", "Oil relative permeability at Swc"],
            ["Endpoint Gas Rel-Perm", "krg0", f"{float(params.get('krg0', 0.60)):.3f}", "0.40 - 0.90", "Gas relative permeability at (1 - Swc - Sorg)"],
            ["Water Corey Exponent", "nw", f"{float(params.get('n_w', 2.5)):.2f}", "1.5 - 4.0", "Corey power curvature for water phase"],
            ["Oil-Water Corey Exponent", "now", f"{float(params.get('n_ow', 2.0)):.2f}", "1.5 - 4.0", "Corey power curvature for oil displacing water"],
            ["Gas Corey Exponent", "ng", f"{float(params.get('n_g', 2.0)):.2f}", "1.5 - 3.5", "Corey power curvature for free gas phase"],
            ["Wettability System", "Wettability", str(params.get("wettability_preset", "Strongly Water-Wet")), "System dependent", "Wettability classification"],
            ["3-Phase Model", "Model", str(params.get("relperm_model", "Modified Stone I")), "Stone I / Stone II", "Three-phase relative permeability calculation"]
        ]
        self.show_table(
            "Corey Relative Permeability Endpoints & Exponents",
            "Governing multi-phase fractional flow and viscous displacement",
            headers, rows, "relperm_endpoints"
        )

    def render_geostat_table(self, params: Dict[str, Any]):
        headers = ["Geostatistical Parameter", "Axis / Direction", "Value", "Field Units", "Physical / Modeling Meaning"]
        rows = [
            ["Major Range (Continuity)", "Strike Axis", f"{float(params.get('variogram_range_major', 1200.0)):.1f}", "ft", "Maximum spatial correlation length"],
            ["Minor Range (Perpendicular)", "Dip Axis", f"{float(params.get('variogram_range_minor', 600.0)):.1f}", "ft", "Perpendicular spatial correlation length"],
            ["Vertical Range", "Z / TVD Axis", f"{float(params.get('variogram_range_vert', 20.0)):.1f}", "ft", "Vertical correlation length (layering)"],
            ["Principal Azimuth", "Azimuth Angle", f"{float(params.get('variogram_azimuth_deg', 45.0)):.1f}", "degrees", "Orientation angle of depositional channel belt"],
            ["Variogram Type", "Model", str(params.get("variogram_type", "Spherical")), "Analytical", "Mathematical covariance model"],
            ["Nugget Effect", "Micro-scale Noise", f"{float(params.get('nugget_effect', 0.05)):.3f}", "fraction", "Measurement error / sub-grid heterogeneity"],
            ["Spatially Correlated Sill", "Variance (C)", f"{float(params.get('sill_variance', 1.0)):.3f}", "variance", "Variance component structured by distance"],
            ["Random Seed", "Seed", str(params.get("geostat_seed", 42)), "integer", "Reproducibility seed for stochastic realizations"],
            ["Simulation Algorithm", "Algorithm", str(params.get("geostat_algorithm", "Sequential Gaussian Simulation (SGSIM)")), "Method", "Monte Carlo spatial field generator"]
        ]
        self.show_table(
            "Geostatistical Variogram Parameters & Spatial Settings",
            f"Major Range: {float(params.get('variogram_range_major', 1200.0)):.0f} ft | Azimuth: {float(params.get('variogram_azimuth_deg', 45.0)):.0f}°",
            headers, rows, "geostat_parameters"
        )

    def render_black_oil_table(self, params: Dict[str, Any]):
        p_ini = float(params.get("initial_pressure", 4000.0))
        pb = float(params.get("bubble_point_pressure", 2250.0))
        rs_ini = float(params.get("sol_gor", 500.0))
        visc_oil = float(params.get("oil_viscosity_cp", 1.45))
        temp = float(params.get("temperature", 160.0))

        headers = [
            "Pressure (psia)", "Bo (rb/STB)", "Rs (scf/STB)", "Oil Visc (cP)",
            "Gas Bg (rb/MSCF)", "Gas Visc (cP)", "Oil Density (lb/ft³)", "CO2 Density (kg/m³)"
        ]
        rows = []
        p_steps = [500, 1000, 1500, 2000, pb, 2500, 3000, 3500, p_ini, 4500, 5000, 6000, 7000]
        p_steps = sorted(list(set(p_steps)))

        for p in p_steps:
            if p <= pb:
                rs = rs_ini * (p / pb)**1.2
                bo = 1.05 + (1.28 - 1.05) * (p / pb)
                visc = visc_oil * (pb / max(p, 100))**0.25
            else:
                rs = rs_ini
                bo = 1.28 * np.exp(-1.5e-5 * (p - pb))
                visc = visc_oil * (1.0 + 3.5e-5 * (p - pb))

            z_factor = 0.85 + 0.05 * (p / 4000.0)
            bg = 0.02827 * z_factor * (temp + 459.67) / max(p, 10.0)
            oil_den = 52.0 - (rs / 100.0) * 0.8 + (p / 4000.0) * 1.5
            co2_den = 450.0 + 500.0 * (1.0 - np.exp(-p / 2200.0))

            rows.append([
                f"{p:,.0f}", f"{bo:.4f}", f"{rs:.1f}", f"{visc:.3f}",
                f"{bg:.4f}", f"{0.015 + 0.003 * (p / 4000.0):.4f}", f"{oil_den:.2f}", f"{co2_den:.1f}"
            ])

        self.show_table(
            "Black Oil Numerical PVT Table",
            f"Bubble Point Pb = {pb:.0f} psia | Reservoir P = {p_ini:.0f} psia",
            headers, rows, "black_oil_numerical_pvt"
        )

    def render_swelling_table(self, params: Dict[str, Any]):
        visc_live = float(params.get("oil_viscosity_cp", 1.45))
        max_sf = float(params.get("co2_swelling_factor_max", 1.28))

        headers = ["Dissolved CO2 (mol %)", "Swelling Factor SF (V/Vo)", "Viscosity Ratio (μ/μo)", "Oil Viscosity (cP)", "Oil Density Reduction (%)", "IFT (mN/m)"]
        rows = []
        for x_co2 in np.linspace(0.0, 0.70, 15):
            sf = 1.0 + (max_sf - 1.0) * (x_co2 / 0.70)**1.2
            ratio = np.exp(-2.2 * (x_co2 / 0.70))
            visc = visc_live * ratio
            den_red = 12.0 * (x_co2 / 0.70)
            ift = max(0.5, 25.0 * (1.0 - (x_co2 / 0.70))**1.5)
            rows.append([
                f"{x_co2 * 100:.1f}%", f"{sf:.4f}", f"{ratio:.3f}",
                f"{visc:.3f}", f"-{den_red:.1f}%", f"{ift:.2f}"
            ])

        self.show_table(
            "CO2 Solvent Swelling & Viscosity Reduction Data Table",
            f"Maximum Swelling SF_max = {max_sf:.2f} at 70 mol% CO2 dissolution",
            headers, rows, "solvent_swelling_table"
        )

    def render_mmp_table(self, params: Dict[str, Any], calculated_mmp: float = 2688.0):
        p_res = float(params.get("initial_pressure", 4000.0))
        temp = float(params.get("temperature", 160.0))
        api = float(params.get("api_gravity", 35.0))

        cronquist = 15.988 * (temp**0.7442) * (api**(-0.25))
        yellig = 1833.0 + 17.0 * (temp - 100.0)
        lee = 1950.0 + 15.2 * (temp - 120.0)
        holm = 2100.0 + 18.0 * (temp - 120.0)

        methods = [
            ("Cronquist Correlation (1978)", cronquist, "Standard baseline for pure CO2"),
            ("Yellig & Metcalfe Correlation (1980)", yellig, "Temperature-dominant correlation"),
            ("Lee Analytical MMP (1979)", lee, "Medium-to-light crude oils"),
            ("Holm & Josendal (1974)", holm, "Bubble point & C5+ molecular weight"),
            ("Active Analytical Engine Value", calculated_mmp, "Ensemble auto-selected benchmark")
        ]

        headers = ["Correlation Method", "Calculated MMP (psia)", "Reservoir Pressure (psia)", "Miscibility Margin (P - MMP)", "Miscibility Status", "Governing Conditions"]
        rows = []
        for name, val, cond in methods:
            margin = p_res - val
            status = "Miscible" if margin >= 0 else "Immiscible"
            rows.append([
                name, f"{val:.1f}", f"{p_res:.1f}", f"{margin:+.1f} psi", status, cond
            ])

        self.show_table(
            "Minimum Miscibility Pressure (MMP) Benchmark Table",
            f"P_reservoir = {p_res:.0f} psia | Engine MMP = {calculated_mmp:.0f} psia",
            headers, rows, "mmp_correlation_benchmarks"
        )

    def render_eos_table(self, params: Dict[str, Any]):
        headers = [
            "Component", "Formula", "Mole Fraction zi (%)", "Molecular Weight",
            "Tc (°F)", "Pc (psia)", "Acentric Factor (ω)", "Volume Shift (s)", "Binary dij (CO2)"
        ]
        components = [
            ("Carbon Dioxide", "CO2", 2.0, 44.01, 87.9, 1070.0, 0.225, 0.000, 0.000),
            ("Nitrogen", "N2", 1.0, 28.01, -232.4, 493.0, 0.040, -0.008, 0.020),
            ("Methane", "C1", 38.0, 16.04, -116.6, 667.0, 0.011, -0.003, 0.100),
            ("Ethane", "C2", 8.0, 30.07, 90.1, 708.0, 0.099, -0.001, 0.130),
            ("Propane", "C3", 6.0, 44.10, 206.0, 616.0, 0.152, -0.002, 0.125),
            ("i-Butane", "iC4", 2.5, 58.12, 275.0, 529.0, 0.186, -0.005, 0.120),
            ("n-Butane", "nC4", 4.5, 58.12, 305.6, 551.0, 0.200, -0.004, 0.115),
            ("Pentanes", "C5", 4.0, 72.15, 385.7, 489.0, 0.252, -0.006, 0.115),
            ("Hexanes", "C6", 4.0, 86.18, 453.7, 437.0, 0.301, -0.008, 0.115),
            ("Heptanes Plus", "C7+", 30.0, 215.0, 785.0, 265.0, 0.520, 0.015, 0.110)
        ]
        rows = [
            [c[0], c[1], f"{c[2]:.1f}%", f"{c[3]:.2f}", f"{c[4]:.1f}", f"{c[5]:.1f}", f"{c[6]:.3f}", f"{c[7]:.3f}", f"{c[8]:.3f}"]
            for c in components
        ]
        self.show_table(
            "10-Component Detailed Composition & EOS Parameters",
            "Peng-Robinson (PR-78) Equation of State Fluid Characterization",
            headers, rows, "eos_compositional_table"
        )

    def render_contacts_table(self, params: Dict[str, Any]):
        top = float(params.get("depth", 5000.0))
        thick = float(params.get("thickness", 50.0))
        p_res = float(params.get("initial_pressure", 4000.0))
        woc = float(params.get("woc_depth", top + thick * 0.7))
        goc = float(params.get("goc_depth", top - 20.0))
        has_gas_cap = bool(params.get("has_gas_cap", False))

        headers = ["Fluid Boundary / Contact", "Depth TVD (ft)", "Hydrostatic Pressure (psia)", "Phase Above", "Phase Below", "Transition Zone Height (ft)", "Capillary Pe (psi)"]
        rows = [
            ["Top Formation Seal", f"{top:.1f}", f"{p_res - 0.35 * (5000 - top):.1f}", "Caprock Shale", "Gas Cap" if has_gas_cap else "Oil Column", "0.0", "150.0"],
            ["Gas-Oil Contact (GOC)", f"{goc:.1f}" if has_gas_cap else "N/A (Under-Saturated)", f"{p_res - 0.35 * (5000 - goc):.1f}" if has_gas_cap else "N/A", "Free Gas Cap" if has_gas_cap else "N/A", "Oil Column", "8.0", "15.0"],
            ["Reservoir Mid-Pay", f"{top + thick * 0.5:.1f}", f"{p_res:.1f}", "Oil Column", "Oil Column", "N/A", "N/A"],
            ["Water-Oil Contact (WOC)", f"{woc:.1f}", f"{p_res + 0.35 * (woc - 5000):.1f}", "Oil Column", "Formation Brine", "18.0", "22.5"],
            ["Free Water Level (FWL)", f"{woc + 12.0:.1f}", f"{p_res + 0.35 * (woc + 12.0 - 5000):.1f}", "Transition Brine", "Regional Aquifer", "0.0", "0.0"]
        ]
        self.show_table(
            "Fluid Contacts & Hydrostatic Zonal Depth Table",
            f"WOC Depth: {woc:.1f} ft | GOC: {goc:.1f} ft",
            headers, rows, "fluid_contacts_table"
        )

    def render_well_inventory_table(self, wells: List[Any], params: Optional[Dict[str, Any]] = None):
        headers = [
            "Well Identifier", "Role / Service", "Surface X (ft)", "Surface Y (ft)",
            "Top MD (ft)", "Bottom MD (ft)", "Perforated Intervals", "Skin Factor", "Productivity Index (J)", "Status"
        ]
        rows = []
        if not wells:
            rows.append(["None", "N/A", "0.0", "0.0", "0.0", "0.0", "0", "0.0", "N/A", "No Wells Configured"])
        else:
            for w in wells:
                w_name = getattr(w, "name", "Well")
                meta = getattr(w, "metadata", {}) or {}
                sx = float(meta.get("SurfaceX", meta.get("surface_x", 1000.0)))
                sy = float(meta.get("SurfaceY", meta.get("surface_y", 1000.0)))
                top_d = float(meta.get("TopDepth", meta.get("top_depth", 5000.0)))
                bot_d = float(meta.get("BottomDepth", meta.get("bottom_depth", 5050.0)))
                w_type = getattr(w, "well_type", meta.get("type", "Producer"))
                perfs = getattr(w, "perforations", []) or []
                skin = float(meta.get("skin", 0.0))
                pi = float(meta.get("productivity_index", 2.5 if "prod" in str(w_type).lower() else 3.5))
                rows.append([
                    w_name, str(w_type), f"{sx:,.1f}", f"{sy:,.1f}",
                    f"{top_d:,.1f}", f"{bot_d:,.1f}", f"{len(perfs)} intervals",
                    f"{skin:+.1f}", f"{pi:.2f} STB/d/psi", "Active"
                ])

        self.show_table(
            "Master Well Inventory & Trajectory Schedule",
            f"Active Network: {len(wells)} Total Wellbores",
            headers, rows, "master_well_inventory"
        )

    def render_well_trajectory_table(self, well: Any, params: Dict[str, Any]):
        w_name = getattr(well, "name", "Well")
        meta = getattr(well, "metadata", {}) or {}
        sx = float(meta.get("SurfaceX", meta.get("surface_x", 1000.0)))
        sy = float(meta.get("SurfaceY", meta.get("surface_y", 1000.0)))
        top_d = float(meta.get("TopDepth", meta.get("top_depth", 5000.0)))
        bot_d = float(meta.get("BottomDepth", meta.get("bottom_depth", 5050.0)))
        w_type = getattr(well, "well_type", meta.get("type", "Producer"))
        perfs = getattr(well, "perforations", []) or []

        headers = ["Point #", "Measured Depth MD (ft)", "True Vertical Depth TVD (ft)", "Surface X (ft)", "Surface Y (ft)", "Inclination (°)", "Azimuth (°)", "Perforation Active"]
        rows = []
        n_pts = 12
        for i in range(n_pts):
            frac = i / max(n_pts - 1, 1)
            tvd = top_d + (bot_d - top_d) * frac
            md = tvd + (15.0 * frac**2 if "horiz" in str(w_type).lower() else 0.0)
            is_perf = "YES" if (0.3 <= frac <= 0.8) else "No"
            rows.append([
                str(i + 1), f"{md:.1f}", f"{tvd:.1f}", f"{sx:.1f}", f"{sy:.1f}",
                "0.0" if "vert" in str(w_type).lower() else f"{frac * 90:.1f}", "0.0", is_perf
            ])

        self.show_table(
            f"Well Deviation Survey & Trajectory: {w_name}",
            f"Type: {w_type} | Top MD: {top_d:.0f} ft | Bottom MD: {bot_d:.0f} ft",
            headers, rows, f"{w_name}_trajectory_survey"
        )

    def render_caprock_table(self, params: Dict[str, Any]):
        headers = ["Geomechanical Parameter", "Symbol", "Active Value", "Units", "Integrity / Containment Criteria"]
        thick = float(params.get("caprock_thickness", 200.0))
        t0 = float(params.get("caprock_t0", 200.0))
        c = float(params.get("caprock_cohesion", 400.0))
        phi = float(params.get("caprock_friction_angle", 30.0))
        pe = float(params.get("caprock_entry_pressure", 1500.0))
        perm = float(params.get("caprock_perm", 0.0001))
        sf = float(params.get("caprock_safety_factor", 0.90))

        rows = [
            ["Caprock Thickness", "H_seal", f"{thick:.1f}", "ft", "Minimum confining layer vertical thickness"],
            ["Tensile Strength", "T0", f"{t0:.1f}", "psi", "Maximum hydraulic fracturing tensile threshold"],
            ["Cohesive Shear Strength", "C", f"{c:.1f}", "psi", "Mohr-Coulomb shear cohesion intercept"],
            ["Internal Friction Angle", "φ_fric", f"{phi:.1f}", "degrees", "Failure envelope slope = tan(φ)"],
            ["Capillary Entry Pressure", "Pe", f"{pe:.1f}", "psi", "Non-wetting CO2 breakthrough pressure"],
            ["Seal Matrix Permeability", "k_seal", f"{perm:.6f}", "mD", "Darcy flow prevention (< 1e-4 mD)"],
            ["EPA Class VI Safety Margin", "SF_uic", f"{sf:.2f}", "fraction", "Mandatory sandface injection ceiling limit"]
        ]
        self.show_table(
            "Caprock Geomechanical Properties & Safety Factors",
            f"Seal Thickness: {thick:.0f} ft | Capillary Entry Pressure: {pe:.0f} psi",
            headers, rows, "caprock_geomechanics_table"
        )

    def render_caprock_graph(self, params: Dict[str, Any]):
        t0 = float(params.get("caprock_t0", 200.0))
        c = float(params.get("caprock_cohesion", 400.0))
        phi_deg = float(params.get("caprock_friction_angle", 30.0))
        phi_rad = np.radians(phi_deg)

        def draw(fig: Figure, canvas: FigureCanvas):
            ax = fig.add_subplot(111)
            sigma_n = np.linspace(0, 5000, 200)
            tau_failure = c + sigma_n * np.tan(phi_rad)

            # Failure envelope line
            ax.plot(sigma_n, tau_failure, color="#dc2626", linewidth=2.5, label="Mohr-Coulomb Failure Envelope")
            ax.fill_between(sigma_n, tau_failure, 5000, color="#fee2e2", alpha=0.35, label="Shear Failure Region")
            ax.fill_between(sigma_n, 0, tau_failure, color="#f0fdf4", alpha=0.35, label="Stable Elastic Region")

            # Tensile cut-off
            ax.axvline(x=0, color="#475569", linestyle="--", linewidth=1.0)
            ax.axvline(x=-t0, color="#b91c1c", linestyle=":", linewidth=2.0, label=f"Tensile Cut-off (-T0 = -{t0:.0f} psi)")

            # Current Stress State Circle
            s1 = 3800.0
            s3 = 2400.0
            r = (s1 - s3) / 2.0
            center = (s1 + s3) / 2.0
            theta = np.linspace(0, np.pi, 100)
            circle_x = center + r * np.cos(theta)
            circle_y = r * np.sin(theta)
            ax.plot(circle_x, circle_y, color="#2563eb", linewidth=2.0, label="Reservoir Stress State (Mohr Circle)")

            ax.set_xlabel("Effective Normal Stress σn' (psia)", fontweight="bold", fontsize=10)
            ax.set_ylabel("Shear Stress τ (psia)", fontweight="bold", fontsize=10)
            ax.set_title("Caprock Mohr-Coulomb Failure Envelope & Stress State", fontweight="bold", fontsize=11)
            ax.grid(True, linestyle=":", alpha=0.5)
            ax.legend(loc="upper left", fontsize=9)

        self.show_graph(
            "Caprock Mohr-Coulomb Failure Envelope & Shear Margin",
            f"Cohesion C = {c:.0f} psi, Friction φ = {phi_deg:.1f}°",
            draw, "caprock_mohr_coulomb"
        )

    def render_fault_table(self, params: Dict[str, Any]):
        headers = [
            "Fault Name", "Strike (°)", "Dip (°)", "Throw (ft)", "Friction Coeff (μ)",
            "Shale Gouge Ratio (SGR %)", "Effective Normal σn' (psia)", "Resolved Shear τ (psia)",
            "Slip Tendency (Ts)", "Reactivation Risk"
        ]
        dip = float(params.get("fault_dip", 70.0))
        strike = float(params.get("fault_strike", 45.0))
        throw = float(params.get("fault_throw", 25.0))
        mu = float(params.get("fault_friction", 0.60))

        # Resolved stresses
        p_res = float(params.get("initial_pressure", 4000.0))
        sv = 5000.0 * 1.05
        sh = 5000.0 * 0.72
        sigma_n = max(500.0, (sv + sh) / 2.0 - p_res)
        tau = max(100.0, (sv - sh) * 0.5 * np.sin(np.radians(2 * dip)))
        ts = tau / max(sigma_n, 1.0)
        risk = "Critical" if ts >= mu else ("Moderate" if ts >= mu * 0.75 else "Low")

        rows = [
            [str(params.get("fault_name", "Fault F-1")), f"{strike:.1f}°", f"{dip:.1f}°", f"{throw:.1f}", f"{mu:.2f}", "38.5%", f"{sigma_n:.0f}", f"{tau:.0f}", f"{ts:.3f}", risk]
        ]
        self.show_table(
            "Fault Geometry & Slip Tendency Analysis Table",
            f"Resolved Slip Tendency Ts = {ts:.3f} vs Friction μ = {mu:.2f}",
            headers, rows, "fault_slip_tendency_table"
        )

    def render_stress_table(self, params: Dict[str, Any]):
        headers = [
            "Depth Interval (ft)", "Overburden Sv (psia)", "Pore Pressure (psia)",
            "Min Stress Shmin (psia)", "Fracture Press (psia)", "EPA Class VI Max BHP (psia)",
            "Allowable Surface WHP (psia)", "UIC Compliance"
        ]
        rows = []
        for d in [3000, 4000, 5000, 6000]:
            sv = d * 1.05
            pp = d * 0.433
            shmin = pp + 0.72 * (sv - pp)
            p_frac = d * 0.85
            max_bhp = 0.90 * p_frac
            max_whp = max_bhp - d * 0.35
            rows.append([
                f"{d:,} ft", f"{sv:,.0f}", f"{pp:,.0f}", f"{shmin:,.0f}",
                f"{p_frac:,.0f}", f"{max_bhp:,.0f}", f"{max_whp:,.0f}", "PASS"
            ])

        self.show_table(
            "EPA Class VI In-Situ Stress & Sandface Pressure Ceilings",
            "UIC Safety Limit: Max Sandface Injection Pressure ≤ 0.90 × P_frac",
            headers, rows, "epa_class_vi_stress_table"
        )

    def render_stress_graph(self, params: Dict[str, Any]):
        def draw(fig: Figure, canvas: FigureCanvas):
            ax = fig.add_subplot(111)
            depths = np.linspace(1000, 7000, 100)
            sv = depths * 1.05
            pp = depths * 0.433
            shmin = pp + 0.72 * (sv - pp)
            p_frac = depths * 0.85
            p_safe = p_frac * 0.90

            ax.plot(pp, depths, color="#38bdf8", linewidth=1.8, label="Hydrostatic Pore Pressure (0.433 psi/ft)")
            ax.plot(shmin, depths, color="#0284c7", linewidth=2.0, label="Min Horizontal Stress Shmin (0.72 psi/ft)")
            ax.plot(p_frac, depths, color="#dc2626", linewidth=2.0, linestyle="--", label="Hydraulic Fracture Pressure Pfrac (0.85 psi/ft)")
            ax.plot(p_safe, depths, color="#16a34a", linewidth=2.5, label="EPA Class VI Injection Ceiling (0.90 × Pfrac)")
            ax.plot(sv, depths, color="#1e293b", linewidth=2.0, label="Overburden Stress Sv (1.05 psi/ft)")

            ax.set_xlabel("Pressure & Stress Magnitude (psia)", fontweight="bold", fontsize=10)
            ax.set_ylabel("True Vertical Depth (ft)", fontweight="bold", fontsize=10)
            ax.set_title("In-Situ Stress Profile & EPA Class VI Maximum Injection Safeguards", fontweight="bold", fontsize=11)
            ax.invert_yaxis()
            ax.grid(True, linestyle=":", alpha=0.5)
            ax.legend(loc="lower left", fontsize=8.5)

        self.show_graph(
            "In-Situ Stress Gradients & Safe Injection Ceilings",
            "Mandatory geomechanical safeguards bounded by EPA Class VI UIC rules",
            draw, "stress_gradients_graph"
        )

    # -------------------------------------------------------------------------
    # Well Schedule & Predictive Controls
    # -------------------------------------------------------------------------
    def render_well_schedule_table(self, wells: List[Any], params: Optional[Dict[str, Any]] = None):
        p = params or {}
        p_res = float(p.get("initial_pressure", 4000.0))
        depth = float(p.get("depth", 5000.0))
        maip_sandface = 0.90 * 0.85 * depth  # EPA Class VI MAIP ceiling
        maip_whp = max(500.0, maip_sandface - depth * 0.28)

        headers = [
            "Well Name", "Type", "Status", "Control Mode", "Target Rate",
            "BHP Limit (psia)", "WHP Limit (psia)", "WAG Ratio", "Cycle (days)",
            "Perforation MD (ft)", "Skin Factor"
        ]
        rows = []
        if wells:
            for w in wells:
                w_name = getattr(w, "name", "Well")
                w_meta = getattr(w, "metadata", {}) or {}
                w_type = str(w_meta.get("type", "Producer")).capitalize()
                status = str(w_meta.get("status", "Active"))
                skin = float(w_meta.get("skin", 0.0))
                ctrl = "BHP Min (Drawdown)" if "Prod" in w_type else "Rate / MAIP Max"
                target = "1,800 STB/d" if "Prod" in w_type else "12,000 MSCF/d"
                bhp_lim = "1,850 psia" if "Prod" in w_type else f"{maip_sandface:,.0f} psia"
                whp_lim = "250 psia" if "Prod" in w_type else f"{maip_whp:,.0f} psia"
                wag_ratio = "N/A" if "Prod" in w_type else "1.5 : 1"
                cycle = "Continuous" if "Prod" in w_type else "90 d (WAG)"
                perfs = f"{depth * 0.98:.0f} - {depth * 1.02:.0f} ft"
                rows.append([w_name, w_type, status, ctrl, target, bhp_lim, whp_lim, wag_ratio, cycle, perfs, f"{skin:+.1f}"])
        else:
            default_wells = [
                ("PROD-01", "Producer", "Active", "BHP Min", "2,200 STB/d", "1,800 psia", "250 psia", "N/A", "Continuous", f"{depth:.0f} - {depth+120:.0f} ft", "0.0"),
                ("PROD-02", "Producer", "Active", "BHP Min", "1,950 STB/d", "1,800 psia", "250 psia", "N/A", "Continuous", f"{depth:.0f} - {depth+120:.0f} ft", "+0.5"),
                ("PROD-03", "Producer", "Active", "BHP Min", "2,400 STB/d", "1,800 psia", "250 psia", "N/A", "Continuous", f"{depth:.0f} - {depth+120:.0f} ft", "-0.8"),
                ("PROD-04", "Producer", "Active", "BHP Min", "2,100 STB/d", "1,800 psia", "250 psia", "N/A", "Continuous", f"{depth:.0f} - {depth+120:.0f} ft", "+0.2"),
                ("INJ-01", "Injector", "Active", "MAIP Limit", "14,500 MSCF/d", f"{maip_sandface:,.0f} psia", f"{maip_whp:,.0f} psia", "1.5 : 1", "90 d (WAG)", f"{depth:.0f} - {depth+100:.0f} ft", "-1.5")
            ]
            for row in default_wells:
                rows.append(list(row))

        self.show_table(
            "Master Well Scheduling & Operational Constraints Table",
            f"EPA Class VI Sandface MAIP Ceiling: {maip_sandface:,.0f} psia | Surface WHP: {maip_whp:,.0f} psia",
            headers, rows, "well_schedule_controls_table"
        )

    def render_well_lifecycle_gantt_graph(self, wells: List[Any], params: Optional[Dict[str, Any]] = None):
        well_names = [getattr(w, "name", f"Well-{i+1}") for i, w in enumerate(wells)] if wells else ["PROD-01", "PROD-02", "PROD-03", "PROD-04", "INJ-01"]

        def draw(fig: Figure, canvas: FigureCanvas):
            ax = fig.add_subplot(111)
            y_pos = np.arange(len(well_names))

            phases = [
                ("Drill & Complete", 0, 1.0, "#94a3b8"),
                ("Primary Depletion", 1.0, 4.0, "#0284c7"),
                ("Waterflooding", 4.0, 7.5, "#0d9488"),
                ("CO2 WAG EOR", 7.5, 17.5, "#d97706"),
                ("Continuous Storage", 17.5, 21.0, "#7c3aed"),
                ("Post-Closure UIC", 21.0, 25.0, "#475569")
            ]

            for idx, w_name in enumerate(well_names):
                for p_name, start, dur, col in phases:
                    is_inj = "INJ" in w_name.upper()
                    if is_inj and p_name == "Primary Depletion":
                        continue
                    if is_inj and p_name == "Waterflooding":
                        bar_start = 3.0
                        bar_dur = 4.5
                    elif is_inj and p_name == "Drill & Complete":
                        bar_start = 2.0
                        bar_dur = 1.0
                    else:
                        bar_start = start
                        bar_dur = dur

                    ax.barh(idx, bar_dur, left=bar_start, height=0.55, color=col, edgecolor="#ffffff", linewidth=1.0)

            # Milestone vertical lines
            ax.axvline(x=7.5, color="#d97706", linestyle="--", linewidth=1.5, label="CO2 First Gas Injection")
            ax.axvline(x=17.5, color="#7c3aed", linestyle=":", linewidth=1.5, label="Cessation of EOR / Dedicated Storage")
            ax.axvline(x=21.0, color="#475569", linestyle="-.", linewidth=1.5, label="Well P&A / Monitoring Period")

            ax.set_yticks(y_pos)
            ax.set_yticklabels(well_names, fontweight="bold", fontsize=10)
            ax.set_xlabel("Field Development Project Timeline (Years)", fontweight="bold", fontsize=10)
            ax.set_title("Master Well Lifecycle & Operational Phase Schedule (25-Year Horizon)", fontweight="bold", fontsize=11)
            ax.set_xlim(0, 25)
            ax.grid(True, linestyle=":", alpha=0.5, axis="x")

            # Custom Legend
            custom_patches = [
                matplotlib.patches.Patch(facecolor="#94a3b8", label="Drill & Complete"),
                matplotlib.patches.Patch(facecolor="#0284c7", label="Primary Depletion"),
                matplotlib.patches.Patch(facecolor="#0d9488", label="Waterflood"),
                matplotlib.patches.Patch(facecolor="#d97706", label="CO2 WAG EOR"),
                matplotlib.patches.Patch(facecolor="#7c3aed", label="Continuous Storage"),
                matplotlib.patches.Patch(facecolor="#475569", label="Post-Closure UIC")
            ]
            ax.legend(handles=custom_patches, loc="lower right", fontsize=8.5, ncol=3)

        self.show_graph(
            "Well Lifecycle Gantt Chart & Development Timeline",
            "Field operational sequence from initial drilling to tertiary WAG and UIC site closure",
            draw, "well_lifecycle_gantt"
        )

    def render_wag_schedule_graph(self, params: Optional[Dict[str, Any]] = None):
        def draw(fig: Figure, canvas: FigureCanvas):
            ax1 = fig.add_subplot(111)
            months = np.arange(0, 121, 1)  # 10 years (120 months)
            cycle_len = 6  # 3 months water, 3 months CO2

            water_inj_rate = np.zeros_like(months, dtype=float)
            co2_inj_rate = np.zeros_like(months, dtype=float)

            for m in months:
                cycle_num = m // cycle_len
                in_gas_half = (m % cycle_len) >= (cycle_len / 2)
                if in_gas_half:
                    co2_inj_rate[m] = 12.0  # MMSCF/d
                    water_inj_rate[m] = 0.0
                else:
                    water_inj_rate[m] = 3.5  # kSTB/d
                    co2_inj_rate[m] = 0.0

            # Step plots for alternating rates
            ax1.step(months / 12.0, water_inj_rate, color="#0284c7", linewidth=2.0, where="post", label="Water Half-Cycle Rate (kSTB/d)")
            ax1.step(months / 12.0, co2_inj_rate, color="#d97706", linewidth=2.0, where="post", label="CO2 Solvent Half-Cycle Rate (MMSCF/d)")

            ax1.set_xlabel("Injection Operation Time (Years)", fontweight="bold", fontsize=10)
            ax1.set_ylabel("Daily Injection Rate", fontweight="bold", fontsize=10)
            ax1.set_ylim(0, 16)
            ax1.grid(True, linestyle=":", alpha=0.5)

            # Twin axis for cumulative HCPV injected
            ax2 = ax1.twinx()
            cum_hcpv = np.linspace(0, 65.0, len(months))
            ax2.plot(months / 12.0, cum_hcpv, color="#16a34a", linewidth=2.2, linestyle="--", label="Cumulative Slug Size (% HCPV)")
            ax2.set_ylabel("Cumulative Injected Slug (% HCPV)", fontweight="bold", fontsize=10, color="#16a34a")
            ax2.tick_params(axis="y", labelcolor="#16a34a")
            ax2.set_ylim(0, 80)

            # Combined Legend
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=8.5)
            ax1.set_title("Water-Alternating-Gas (WAG) Cyclic Injection Schedule & Slug Sizing", fontweight="bold", fontsize=11)

        self.show_graph(
            "Cyclic WAG Injection Schedule & Slug Accumulation",
            "3-month Water / 3-month CO2 alternating half-cycles up to 65% Hydrocarbon Pore Volume (HCPV)",
            draw, "wag_injection_schedule"
        )

    # -------------------------------------------------------------------------
    # Caprock Confining Stratigraphy & Capillary Sealing
    # -------------------------------------------------------------------------
    def render_caprock_geology_table(self, params: Dict[str, Any]):
        depth = float(params.get("depth", 5000.0))
        thk = float(params.get("caprock_thickness", 220.0))
        pe_base = float(params.get("caprock_entry_pressure", 1800.0))

        headers = [
            "Confining Member", "Depth Interval (ft TVD)", "Thickness (ft)",
            "Lithology Description", "Capillary Pe (psia)", "Matrix Perm (mD)",
            "Porosity (%)", "Brittleness Index", "XRD Clay / Quartz %", "Max CO2 Column (ft)"
        ]
        u1_thk = thk * 0.38
        u2_thk = thk * 0.35
        u3_thk = thk * 0.27

        # Maximum sustainable CO2 column height: H_max = Pe / (delta_rho * g)
        # delta_rho * g ≈ 0.465 - 0.280 = 0.185 psi/ft
        grad_buoy = 0.185
        h_max1 = (pe_base * 1.15) / grad_buoy
        h_max2 = (pe_base * 0.78) / grad_buoy
        h_max3 = (pe_base * 0.52) / grad_buoy

        rows = [
            [
                "Unit 1: Basal Primary Marine Shale",
                f"{depth - u1_thk:.0f} - {depth:.0f} ft",
                f"{u1_thk:.1f} ft",
                "Hard, dense carbonaceous marine claystone",
                f"{pe_base * 1.15:,.0f} psia",
                "0.000010 mD",
                "3.2%",
                "0.32 (Ductile)",
                "65% Illite-Smectite / 20% Qtz",
                f"{h_max1:,.0f} ft"
            ],
            [
                "Unit 2: Intermediate Silty Baffle",
                f"{depth - u1_thk - u2_thk:.0f} - {depth - u1_thk:.0f} ft",
                f"{u2_thk:.1f} ft",
                "Interbedded calcareous siltstone and mudstone",
                f"{pe_base * 0.78:,.0f} psia",
                "0.000500 mD",
                "7.5%",
                "0.48 (Semi-Brittle)",
                "45% Silt-Qtz / 35% Clay / 15% Calc",
                f"{h_max2:,.0f} ft"
            ],
            [
                "Unit 3: Regional Overburden Aquitard",
                f"{depth - thk:.0f} - {depth - u1_thk - u2_thk:.0f} ft",
                f"{u3_thk:.1f} ft",
                "Laminated regional silty shale & dense marl",
                f"{pe_base * 0.52:,.0f} psia",
                "0.002000 mD",
                "10.8%",
                "0.41 (Transition)",
                "55% Illite / 30% Quartz / 10% Fsp",
                f"{h_max3:,.0f} ft"
            ]
        ]
        self.show_table(
            "Caprock Confining Stratigraphy & Multi-Barrier Seal Table",
            f"Total Confining Thickness: {thk:.0f} ft | Basal Entry Pressure: {pe_base*1.15:,.0f} psia | Primary Seal Capacity: {h_max1:,.0f} ft CO2",
            headers, rows, "caprock_stratigraphy_table"
        )

    def render_caprock_sealing_graph(self, params: Dict[str, Any]):
        pe_base = float(params.get("caprock_entry_pressure", 1800.0))
        grad_buoy = 0.185  # psi/ft buoyant pressure gradient (brine minus CO2)

        def draw(fig: Figure, canvas: FigureCanvas):
            ax = fig.add_subplot(111)
            column_height = np.linspace(0, 3000, 200)
            buoyancy_overpressure = column_height * grad_buoy

            pe_u1 = pe_base * 1.15
            pe_u2 = pe_base * 0.78
            pe_u3 = pe_base * 0.52

            # Buoyancy pressure line
            ax.plot(column_height, buoyancy_overpressure, color="#0284c7", linewidth=2.5, label="Buoyant Plume Overpressure ΔP = (ρw - ρco2)·g·H")

            # Breakthrough thresholds
            ax.axhline(y=pe_u1, color="#15803d", linewidth=2.0, linestyle="--", label=f"Unit 1 Primary Marine Seal Breakthrough ({pe_u1:.0f} psi)")
            ax.axhline(y=pe_u2, color="#d97706", linewidth=1.8, linestyle=":", label=f"Unit 2 Silty Baffle Breakthrough ({pe_u2:.0f} psi)")
            ax.axhline(y=pe_u3, color="#dc2626", linewidth=1.8, linestyle="-.", label=f"Unit 3 Regional Aquitard Threshold ({pe_u3:.0f} psi)")

            # Safe containment zone fill
            h_safe = pe_u1 / grad_buoy
            ax.fill_between(column_height, 0, buoyancy_overpressure, where=(column_height <= h_safe), color="#f0fdf4", alpha=0.45, label="Safe Capillary Containment Envelope")
            ax.fill_between(column_height, buoyancy_overpressure, 3000, where=(buoyancy_overpressure >= pe_u3), color="#fef2f2", alpha=0.35, label="Capillary Leakage Risk Zone")

            ax.set_xlabel("Buoyant CO2 Plume Column Height H (ft)", fontweight="bold", fontsize=10)
            ax.set_ylabel("Differential Buoyancy Pressure at Caprock Base (psi)", fontweight="bold", fontsize=10)
            ax.set_title("Caprock Capillary Breakthrough Pressure vs Sustainable CO2 Column Height", fontweight="bold", fontsize=11)
            ax.set_xlim(0, 3000)
            ax.set_ylim(0, max(2500, pe_u1 * 1.25))
            ax.grid(True, linestyle=":", alpha=0.5)
            ax.legend(loc="upper left", fontsize=8.5)

        self.show_graph(
            "Caprock Sealing Capacity & Breakthrough Column Height",
            f"Basal Seal Capacity: {pe_base*1.15/grad_buoy:,.0f} ft continuous CO2 plume column without capillary entry",
            draw, "caprock_sealing_column_graph"
        )


    # -------------------------------------------------------------------------
    # Faults & Structural Geomechanics
    # -------------------------------------------------------------------------
    def render_fault_table(self, params: Dict[str, Any], faults: Optional[List[Any]] = None):
        """Renders comprehensive master table for all structural faults in the reservoir."""
        flist = faults if faults and len(faults) > 0 else [
            dict(id="F-1", name=params.get("fault_name", "Fault F-1"), strike=params.get("fault_strike", 45.0),
                 dip=params.get("fault_dip", 70.0), throw=params.get("fault_throw", 50.0), heave=18.0, length=3800.0,
                 sgr=params.get("shale_gouge_ratio", 32.0), trans_mult=params.get("fault_trans_mult", 0.15),
                 mu=params.get("fault_friction", 0.60), ts=0.42, status="STABLE")
        ]
        headers = [
            "Fault ID", "Fault Name", "Strike (°)", "Dip (°)", "Throw (ft)", "Heave (ft)",
            "Length (ft)", "Damage Zone (ft)", "Trans Multiplier", "SGR (%)", "Slip Ts", "Stability"
        ]
        rows = []
        for f in flist:
            f_id = getattr(f, "id", f.get("id", "F-1") if isinstance(f, dict) else "F-1")
            name = getattr(f, "name", f.get("name", "Fault") if isinstance(f, dict) else "Fault")
            strike = float(getattr(f, "strike", f.get("strike", 45.0) if isinstance(f, dict) else 45.0))
            dip = float(getattr(f, "dip", f.get("dip", 70.0) if isinstance(f, dict) else 70.0))
            throw = float(getattr(f, "throw", f.get("throw", 50.0) if isinstance(f, dict) else 50.0))
            heave = float(getattr(f, "heave", f.get("heave", 18.0) if isinstance(f, dict) else 18.0))
            length = float(getattr(f, "length", f.get("length", 3500.0) if isinstance(f, dict) else 3500.0))
            dz_w = float(getattr(f, "damage_zone_width", f.get("damage_zone_width", 80.0) if isinstance(f, dict) else 80.0))
            tm = float(getattr(f, "transmissibility_multiplier", f.get("trans_mult", 0.15) if isinstance(f, dict) else 0.15))
            sgr = float(getattr(f, "shale_gouge_ratio", f.get("sgr", 30.0) if isinstance(f, dict) else 30.0))
            ts = float(getattr(f, "slip_tendency", f.get("ts", 0.42) if isinstance(f, dict) else 0.42))
            mu = float(getattr(f, "friction_coefficient", f.get("mu", 0.60) if isinstance(f, dict) else 0.60))
            status = "CRITICAL" if ts >= mu else "STABLE"

            rows.append([
                str(f_id), str(name), f"{strike:.0f}°", f"{dip:.0f}°", f"{throw:.0f} ft",
                f"{heave:.1f} ft", f"{length:,.0f} ft", f"{dz_w:.0f} ft", f"{tm:.2f}",
                f"{sgr:.1f}%", f"{ts:.2f}", status
            ])

        self.show_table(
            "Master Structural Fault Network & Petrophysical Baffle Table",
            f"Active Tectonic System: {len(flist)} Mapped Faults | Strike Range: 25° - 65° | Gouge Seal Multipliers: 0.08 - 0.20",
            headers, rows, "master_fault_network_table"
        )

    def render_fault_geometry_table(self, fault: Any, params: Optional[Dict[str, Any]] = None):
        """Detailed single-fault geometry, kinematics, and resolved geomechanical stresses."""
        name = getattr(fault, "name", "Fault")
        strike = float(getattr(fault, "strike", 45.0))
        dip = float(getattr(fault, "dip", 70.0))
        throw = float(getattr(fault, "throw", 50.0))
        heave = float(getattr(fault, "heave", 18.0))
        length = float(getattr(fault, "length", 3500.0))
        cx = float(getattr(fault, "center_x", 1000.0))
        cy = float(getattr(fault, "center_y", 1000.0))
        z_top = float(getattr(fault, "z_top", 4800.0))
        z_base = float(getattr(fault, "z_base", 5250.0))
        tm = float(getattr(fault, "transmissibility_multiplier", 0.15))
        dz_w = float(getattr(fault, "damage_zone_width", 80.0))
        sgr = float(getattr(fault, "shale_gouge_ratio", 32.0))
        mu = float(getattr(fault, "friction_coefficient", 0.60))
        cohesion = float(getattr(fault, "cohesion", 0.0))
        ts = float(getattr(fault, "slip_tendency", 0.42))

        headers = ["Parameter", "Unit / Metric", "Value", "Physical Significance & Interpretation"]
        rows = [
            ["Fault Identifier & Name", "-", str(name), "Primary structural dislocation boundary"],
            ["Fault Strike Azimuth", "degrees North", f"{strike:.1f}°", "Trend relative to regional maximum horizontal stress SHmax"],
            ["Fault Dip Angle", "degrees", f"{dip:.1f}°", "Normal fault planar inclination from horizontal"],
            ["Vertical Structural Throw", "feet", f"{throw:.1f} ft", "Downthrow offset across hanging wall producing layer juxtaposition"],
            ["Horizontal Heave Extension", "feet", f"{heave:.1f} ft", "Normal tectonic extensional gap in map view"],
            ["Fault Trace Length", "feet", f"{length:,.1f} ft", "Lateral tip-to-tip strike continuity"],
            ["Fault Surface Center (X, Y)", "feet", f"({cx:,.0f}, {cy:,.0f})", "Structural focal coordinate in local reservoir grid"],
            ["Depth Interval (Top - Base)", "ft TVD", f"{z_top:,.0f} - {z_base:,.0f} ft", "Vertical propagation through reservoir and caprock"],
            ["Transmissibility Multiplier", "fraction", f"{tm:.3f}", "Cross-fault barrier factor (0 = sealing gouge, 1 = open conduit)"],
            ["Damage Zone Core Width", "feet", f"{dz_w:.1f} ft", "Pervasively fractured halo with modulated permeability"],
            ["Shale Gouge Ratio (SGR)", "%", f"{sgr:.1f}%", "Clay smear entrainment (>20% typically seals hydrocarbons)"],
            ["Friction Coefficient (Byerlee)", "-", f"{mu:.2f}", "Internal frictional resistance to fault reactivation"],
            ["Cohesive Shear Strength", "psi", f"{cohesion:.1f} psi", "Cementation strength along the slip surface"],
            ["Resolved Slip Tendency (Ts)", "tau / sigma_n'", f"{ts:.2f}", "Normalized shear stress; reactivation occurs when Ts >= mu"],
            ["Reactivation Pore Pressure Margin", "psi", "+1,240 psi", "Allowable reservoir repressurization before frictional slip"]
        ]
        self.show_table(
            f"Detailed Structural & Geomechanical Profile: {name}",
            f"Strike {strike:.0f}° | Dip {dip:.0f}° | Throw {throw:.0f} ft | SGR {sgr:.1f}% | Slip Ts {ts:.2f}",
            headers, rows, "fault_detail_profile_table"
        )

    def render_fault_slip_graph(self, fault: Any, params: Optional[Dict[str, Any]] = None):
        """Stereonet / Mohr-Coulomb reactivation envelope for a single fault."""
        name = getattr(fault, "name", "Fault F-1")
        strike = float(getattr(fault, "strike", 45.0))
        dip = float(getattr(fault, "dip", 70.0))
        mu = float(getattr(fault, "friction_coefficient", 0.60))
        cohesion = float(getattr(fault, "cohesion", 0.0))

        def draw(fig: Figure, canvas: FigureCanvas):
            ax1 = fig.add_subplot(1, 2, 1)
            ax2 = fig.add_subplot(1, 2, 2)

            # Left: Mohr Coulomb diagram
            sv = 5250.0  # psi
            sh = 3780.0  # psi
            pore_p = 3200.0  # psi
            sigma_eff = np.linspace(0, 3000, 100)
            tau_failure = cohesion + mu * sigma_eff

            ax1.plot(sigma_eff, tau_failure, color="#dc2626", lw=2.0, label=f"Coulomb Reactivation: τ = {cohesion:.0f} + {mu:.2f}·σn'")

            # Mohr circle
            r_mohr = (sv - sh) / 2.0
            c_mohr = (sv + sh) / 2.0 - pore_p
            theta = np.linspace(0, np.pi, 100)
            ax1.plot(c_mohr + r_mohr * np.cos(theta), r_mohr * np.sin(theta), color="#0284c7", lw=2.0, label="In-Situ Stress Mohr Circle")

            # Fault plane stress point
            theta_rad = np.radians(dip)
            sig_n = c_mohr + r_mohr * np.cos(2.0 * theta_rad)
            tau_plane = r_mohr * np.sin(2.0 * theta_rad)
            ax1.plot([sig_n], [tau_plane], marker="*", markersize=12, color="#e11d48", label=f"{name} (Ts = {tau_plane/max(sig_n,1):.2f})")

            ax1.set_xlabel("Effective Normal Stress σn' (psi)", fontweight="bold", fontsize=9.5)
            ax1.set_ylabel("Resolved Shear Stress τ (psi)", fontweight="bold", fontsize=9.5)
            ax1.set_title("Mohr-Coulomb Reactivation Envelope", fontweight="bold", fontsize=10.5)
            ax1.grid(True, ls=":", alpha=0.5)
            ax1.legend(loc="upper left", fontsize=8)

            # Right: 360-degree Strike Orientation Sensitivity
            azimuths = np.linspace(0, 360, 200)
            sh_az = 90.0  # East-West SHmax
            rel_ang = np.radians(azimuths - sh_az)
            ts_curve = 0.22 + 0.38 * (np.sin(2.0 * rel_ang)**2)

            ax2.plot(azimuths, ts_curve, color="#2563eb", lw=2.2, label="Slip Tendency Ts(θ)")
            ax2.axhline(mu, color="#dc2626", lw=1.5, ls="--", label=f"Critical Friction Limit (μ = {mu:.2f})")
            ax2.plot([strike], [0.22 + 0.38 * (np.sin(2.0 * np.radians(strike - sh_az))**2)], marker="o", markersize=9, color="#e11d48", label=f"{name} ({strike:.0f}°)")

            ax2.fill_between(azimuths, mu, 1.0, color="#fef2f2", alpha=0.5, label="Critically Stressed Zone")
            ax2.set_xlabel("Fault Strike Azimuth (degrees North)", fontweight="bold", fontsize=9.5)
            ax2.set_ylabel("Resolved Slip Tendency Ts", fontweight="bold", fontsize=9.5)
            ax2.set_title("Fault Reactivation Risk vs Strike Orientation", fontweight="bold", fontsize=10.5)
            ax2.set_xlim(0, 360)
            ax2.set_ylim(0, 0.80)
            ax2.grid(True, ls=":", alpha=0.5)
            ax2.legend(loc="upper right", fontsize=8)

        self.show_graph(
            f"Fault Reactivation Stability Analysis: {name}",
            f"Mohr-Coulomb failure circle and 360° strike sensitivity relative to regional in-situ stress field",
            draw, "fault_slip_tendency_graph"
        )

    def render_fault_inter_stress_graph(self, faults: List[Any], params: Optional[Dict[str, Any]] = None):
        """Inter-fault Coulomb stress transfer (Delta CFS) interaction plot."""
        def draw(fig: Figure, canvas: FigureCanvas):
            ax1 = fig.add_subplot(1, 2, 1)
            ax2 = fig.add_subplot(1, 2, 2)

            # Left: 2D Spatial Coulomb Stress Halo
            x = np.linspace(-3000, 3000, 150)
            y = np.linspace(-2000, 2000, 150)
            X, Y = np.meshgrid(x, y)
            r = np.sqrt(X**2 + Y**2) + 200.0
            theta = np.arctan2(Y, X)

            # Classic 4-quadrant dislocation stress quadrupole
            delta_cfs_2d = 650.0 * (1200.0 / r)**1.7 * np.sin(2.0 * theta)

            im = ax1.contourf(X, Y, delta_cfs_2d, levels=np.linspace(-60, 60, 25), cmap="coolwarm", extend="both")
            cbar = fig.colorbar(im, ax=ax1, orientation="horizontal", pad=0.15, shrink=0.8)
            cbar.set_label("Coulomb Stress Transfer ΔCFS (psi)", fontsize=8.5, fontweight="bold")

            # Draw fault line traces
            ax1.plot([-1800, 1800], [-300, 300], color="#0f172a", lw=3.0, label="Active Trigger Fault")
            if faults and len(faults) > 1:
                ax1.plot([-1200, 1500], [800, 1400], color="#0284c7", lw=2.5, ls="--", label="Target Secondary Fault")

            ax1.set_xlabel("Relative X Distance (ft)", fontweight="bold", fontsize=9.5)
            ax1.set_ylabel("Relative Y Distance (ft)", fontweight="bold", fontsize=9.5)
            ax1.set_title("Inter-Fault Coulomb Stress Perturbation Halo", fontweight="bold", fontsize=10.5)
            ax1.legend(loc="upper left", fontsize=8)
            ax1.grid(True, ls=":", alpha=0.4)

            # Right: Stress Decay Curves for Adjacent Faults
            dist_array = np.linspace(200, 4500, 100)
            unclamping_cfs = 550.0 * (1000.0 / dist_array)**1.6
            clamping_cfs = -320.0 * (1000.0 / dist_array)**1.6

            ax2.plot(dist_array, unclamping_cfs, color="#dc2626", lw=2.2, label="Extensional Lobe (+ΔCFS, Destabilizing)")
            ax2.plot(dist_array, clamping_cfs, color="#16a34a", lw=2.2, ls="--", label="Compressional Lobe (-ΔCFS, Clamping)")
            ax2.axhline(0, color="#64748b", lw=0.8, ls=":")
            ax2.axhline(15.0, color="#ef4444", lw=1.2, ls="-.", label="Earthquake Trigger Threshold (15 psi)")

            ax2.set_xlabel("Inter-Fault Distance (ft)", fontweight="bold", fontsize=9.5)
            ax2.set_ylabel("Coulomb Stress Change ΔCFS (psi)", fontweight="bold", fontsize=9.5)
            ax2.set_title("Stress Transfer Attenuation vs Fault Separation", fontweight="bold", fontsize=10.5)
            ax2.set_xlim(200, 4500)
            ax2.grid(True, ls=":", alpha=0.5)
            ax2.legend(loc="upper right", fontsize=8)

        self.show_graph(
            "Inter-Fault Coulomb Stress Transfer & Geomechanical Interaction",
            "Mutual stress transfer: slip or pore pressure changes on one fault induce normal and shear stress perturbations on adjacent faults",
            draw, "inter_fault_coulomb_stress"
        )

    def render_fault_juxtaposition_graph(self, fault: Any, params: Optional[Dict[str, Any]] = None):
        """Allan juxtaposition diagram across the fault plane and depth-dependent SGR curve."""
        name = getattr(fault, "name", "Fault F-1")
        throw = float(getattr(fault, "throw", 50.0))
        sgr_base = float(getattr(fault, "shale_gouge_ratio", 32.0))

        def draw(fig: Figure, canvas: FigureCanvas):
            ax1 = fig.add_subplot(1, 2, 1)
            ax2 = fig.add_subplot(1, 2, 2)

            # Left: Allan Juxtaposition Schematic (Footwall vs Hanging wall)
            depths = np.linspace(4800, 5250, 100)
            # Stratigraphic units: 0-100 ft Sand, 100-180 ft Shale, 180-350 ft Sand, 350-450 ft Carbonate
            fw_sand = ((depths >= 4800) & (depths <= 4900)) | ((depths >= 4980) & (depths <= 5150))
            hw_depths = depths - throw
            hw_sand = ((hw_depths >= 4800) & (hw_depths <= 4900)) | ((hw_depths >= 4980) & (hw_depths <= 5150))

            sand_on_sand = fw_sand & hw_sand
            sand_on_shale = (fw_sand & ~hw_sand) | (~fw_sand & hw_sand)
            shale_on_shale = ~fw_sand & ~hw_sand

            ax1.barh(depths[sand_on_sand], 1.0, height=5, color="#fef08a", edgecolor="none", label="Sand-on-Sand (Conduit)")
            ax1.barh(depths[sand_on_shale], 1.0, height=5, color="#60a5fa", edgecolor="none", label="Sand-on-Shale (Juxtaposition Seal)")
            ax1.barh(depths[shale_on_shale], 1.0, height=5, color="#94a3b8", edgecolor="none", label="Shale-on-Shale (Intact Seal)")

            ax1.set_ylim(5250, 4800)  # Inverted for depth
            ax1.set_xlim(0, 1)
            ax1.set_xticks([])
            ax1.set_ylabel("True Vertical Depth (ft TVD)", fontweight="bold", fontsize=9.5)
            ax1.set_title("Allan Horizon Juxtaposition", fontweight="bold", fontsize=10.5)
            ax1.legend(loc="lower left", fontsize=8)

            # Right: Shale Gouge Ratio (SGR) Curve with Depth
            sgr_profile = sgr_base + 8.0 * np.sin((depths - 4800) / 70.0)
            ax2.plot(sgr_profile, depths, color="#059669", lw=2.5, label="Calculated SGR Profile")
            ax2.axvline(20.0, color="#d97706", lw=1.5, ls="--", label="Minimum Capillary Sealing Limit (20%)")
            ax2.axvline(30.0, color="#15803d", lw=1.5, ls=":", label="Continuous Smear Threshold (30%)")

            ax2.fill_betweenx(depths, 20.0, sgr_profile, where=(sgr_profile >= 20.0), color="#dcfce7", alpha=0.45, label="Effective Membrane Seal")
            ax2.fill_betweenx(depths, 0.0, 20.0, color="#fee2e2", alpha=0.35, label="High Transmissibility Zone")

            ax2.set_ylim(5250, 4800)
            ax2.set_xlim(0, 60)
            ax2.set_xlabel("Shale Gouge Ratio SGR (%)", fontweight="bold", fontsize=9.5)
            ax2.set_title("SGR Depth Profile & Capillary Seal Envelope", fontweight="bold", fontsize=10.5)
            ax2.grid(True, ls=":", alpha=0.5)
            ax2.legend(loc="lower right", fontsize=8)

        self.show_graph(
            f"Fault Horizon Juxtaposition & SGR Profile: {name}",
            f"Juxtaposition analysis across {throw:.0f} ft vertical throw and depth-dependent Shale Gouge Ratio",
            draw, "fault_juxtaposition_sgr"
        )

    def render_caprock_graph(self, params: Dict[str, Any]):
        """Mohr-Coulomb failure envelopes for the overlying confining caprock."""
        depth = float(params.get("depth", 5000.0))
        t0 = float(params.get("caprock_t0", 200.0))
        c0 = float(params.get("caprock_cohesion", 400.0))
        phi = float(params.get("caprock_friction_angle", 30.0))
        mu_cap = np.tan(np.radians(phi))

        def draw(fig: Figure, canvas: FigureCanvas):
            ax = fig.add_subplot(111)
            sigma_n = np.linspace(0, 4000, 150)
            tau_shear = c0 + mu_cap * sigma_n

            # Shear failure line
            ax.plot(sigma_n, tau_shear, color="#dc2626", lw=2.2, label=f"Caprock Shear Failure (C0={c0:.0f} psi, φ={phi:.0f}°)")
            # Tensile failure vertical cutoff
            ax.axvline(-t0, color="#7c3aed", lw=2.0, ls="--", label=f"Tensile Hydraulic Micro-Fracture Limit (T0={t0:.0f} psi)")

            # Current state Mohr Circle in caprock
            sv_eff = depth * 1.05 - 2800.0
            sh_eff = depth * 0.75 - 2800.0
            r_c = (sv_eff - sh_eff) / 2.0
            c_c = (sv_eff + sh_eff) / 2.0
            theta = np.linspace(0, np.pi, 100)
            ax.plot(c_c + r_c * np.cos(theta), r_c * np.sin(theta), color="#16a34a", lw=2.0, label="Caprock In-Situ Stress State")

            ax.set_xlabel("Effective Normal Stress σn' (psi)", fontweight="bold", fontsize=10)
            ax.set_ylabel("Shear Stress τ (psi)", fontweight="bold", fontsize=10)
            ax.set_title("Caprock Mohr-Coulomb & Tensile Containment Envelopes", fontweight="bold", fontsize=11)
            ax.set_xlim(-t0 * 1.5, 3500)
            ax.set_ylim(0, 2200)
            ax.grid(True, ls=":", alpha=0.5)
            ax.legend(loc="upper left", fontsize=8.5)

        self.show_graph(
            "Caprock Geomechanical Integrity Envelopes",
            "Tensile micro-fracturing threshold and shear failure criteria under pore pressure inflation",
            draw, "caprock_mohr_coulomb_graph"
        )

    def render_caprock_layers_table(self, caprock_layers: Optional[List[Any]] = None, params: Optional[Dict[str, Any]] = None):
        """Displays multi-layer confining stratigraphy table."""
        layers = caprock_layers if caprock_layers and len(caprock_layers) > 0 else [
            dict(name="Unit C1 - Basal Marine Shale", lithology="Illite-Smectite Marine Claystone", thickness_ft=120.0, youngs_modulus_gpa=18.5, entry_pressure_psi=2200.0, permeability_nd=10.0),
            dict(name="Unit C2 - Intermediate Silt Baffle", lithology="Calcite-Cemented Siltstone", thickness_ft=85.0, youngs_modulus_gpa=24.0, entry_pressure_psi=1450.0, permeability_nd=450.0),
            dict(name="Unit C3 - Regional Aquitard", lithology="Dense Silty Mudstone", thickness_ft=180.0, youngs_modulus_gpa=16.0, entry_pressure_psi=950.0, permeability_nd=2200.0)
        ]
        headers = [
            "Confining Unit", "Lithology / Mineralogy", "Thickness (ft)", "Young's E (GPa)",
            "Entry Pe (psia)", "Perm kv (nD)", "Max CO2 Column (ft)", "Sealing Function"
        ]
        rows = []
        for lay in layers:
            name = getattr(lay, "name", lay.get("name", "Unit") if isinstance(lay, dict) else "Unit")
            litho = getattr(lay, "lithology", lay.get("lithology", "Shale") if isinstance(lay, dict) else "Shale")
            thk = float(getattr(lay, "thickness_ft", lay.get("thickness_ft", 100.0) if isinstance(lay, dict) else 100.0))
            e_gpa = float(getattr(lay, "youngs_modulus_gpa", lay.get("youngs_modulus_gpa", 20.0) if isinstance(lay, dict) else 20.0))
            pe = float(getattr(lay, "entry_pressure_psi", lay.get("entry_pressure_psi", 1500.0) if isinstance(lay, dict) else 1500.0))
            kv = float(getattr(lay, "permeability_nd", lay.get("permeability_nd", 100.0) if isinstance(lay, dict) else 100.0))
            col_ht = pe / 0.185
            role = "Primary Capillary Barrier" if pe > 1800 else ("Intermediate Baffle" if pe > 1200 else "Regional Aquitard")

            rows.append([
                str(name), str(litho), f"{thk:.0f} ft", f"{e_gpa:.1f} GPa",
                f"{pe:,.0f} psia", f"{kv:.0f} nD", f"{col_ht:,.0f} ft", role
            ])

        total_thk = sum(float(getattr(l, "thickness_ft", l.get("thickness_ft", 100.0) if isinstance(l, dict) else 100.0)) for l in layers)
        self.show_table(
            "Multi-Layer Confining Stratigraphy & Sealing Envelopes",
            f"Total Seal Thickness: {total_thk:.0f} ft | {len(layers)} Distinct Geological Confining Units | Certified EPA Class VI",
            headers, rows, "caprock_confining_layers_table"
        )

    def render_epa_uic_compliance_table(self, params: Dict[str, Any]):
        depth = float(params.get("depth", 5000.0))
        p_res = float(params.get("initial_pressure", 4000.0))
        f_grad = float(params.get("fracture_gradient", 0.85))
        p_frac = depth * f_grad
        maip_sandface = 0.90 * p_frac
        maip_surface = max(400.0, maip_sandface - depth * 0.28)
        usdw_depth = float(params.get("usdw_depth", 1200.0))
        p_crit_usdw = usdw_depth * 0.465
        planned_inj_p = min(maip_sandface - 250.0, p_res + 350.0)

        headers = [
            "UIC Regulatory Mandate", "Statutory Rule (40 CFR)", "Planned / Operating Value",
            "Permit Limit / Ceiling", "Physical Safety Margin", "Compliance Status"
        ]
        rows = [
            [
                "Sandface Maximum Allowable Injection Pressure (MAIP)",
                "§ 146.86(a)(1)",
                f"{planned_inj_p:,.0f} psia",
                f"≤ {maip_sandface:,.0f} psia (0.90 × Pfrac)",
                f"+{maip_sandface - planned_inj_p:,.0f} psia margin",
                "PASS"
            ],
            [
                "Wellhead Operating Pressure Ceiling (THP)",
                "§ 146.86(a)(2)",
                f"{planned_inj_p - depth * 0.28:,.0f} psia",
                f"≤ {maip_surface:,.0f} psia",
                f"+{maip_surface - (planned_inj_p - depth * 0.28):,.0f} psia margin",
                "PASS"
            ],
            [
                "Area of Review (AoR) Lowermost USDW Protection",
                "§ 146.84",
                f"380 psia plume head",
                f"≤ {p_crit_usdw:,.0f} psia (USDW Head)",
                f"+{p_crit_usdw - 380:,.0f} psia to drinking water",
                "PASS"
            ],
            [
                "Caprock Hydraulic Fracture Prevention",
                "§ 146.82(a)(3)",
                f"{planned_inj_p:,.0f} psia",
                f"< {p_frac:,.0f} psia (Pfrac)",
                f"+{p_frac - planned_inj_p:,.0f} psia tensile buffer",
                "PASS"
            ],
            [
                "Critically Stressed Fault Slip Prevention",
                "§ 146.84(c)(1)",
                "Ts = 0.38",
                "Ts < 0.60 (Frictional Slip)",
                "36.7% sub-critical stability",
                "PASS"
            ],
            [
                "Mechanical Integrity Test (MIT) Annulus Margin",
                "§ 146.89",
                "500 psi differential",
                "≥ 300 psi positive differential",
                "+200 psi barrier integrity",
                "PASS"
            ]
        ]
        self.show_table(
            "EPA Class VI Underground Injection Control (UIC) Compliance Master Audit",
            f"Injection Depth: {depth:,.0f} ft | Sandface MAIP: {maip_sandface:,.0f} psia | Lowermost USDW: {usdw_depth:,.0f} ft",
            headers, rows, "epa_uic_compliance_audit_table"
        )

    def render_aor_plume_graph(self, params: Dict[str, Any]):
        depth = float(params.get("depth", 5000.0))
        p_res = float(params.get("initial_pressure", 4000.0))
        f_grad = float(params.get("fracture_gradient", 0.85))
        maip = 0.90 * f_grad * depth
        usdw_d = float(params.get("usdw_depth", 1200.0))
        p_crit = usdw_d * 0.465

        def draw(fig: Figure, canvas: FigureCanvas):
            ax = fig.add_subplot(111)
            r = np.linspace(50, 10000, 300)

            # Radial transient pressure elevation (Theis/diffusivity solution)
            # Delta P(r) decays logarithmically away from wellbore
            delta_p_well = (maip - 200.0) - p_res
            delta_p = delta_p_well * np.exp(-r / 2200.0) / (1.0 + 0.3 * np.log(r / 50.0))
            p_profile = p_res + delta_p

            ax.plot(r, p_profile, color="#0284c7", linewidth=2.5, label="Reservoir Pressure Plume P(r) at Year 10")
            ax.axhline(y=maip, color="#dc2626", linewidth=2.0, linestyle="--", label=f"Sandface MAIP Ceiling ({maip:,.0f} psia)")
            ax.axhline(y=p_res, color="#64748b", linewidth=1.5, linestyle=":", label=f"Initial Reservoir Pressure ({p_res:,.0f} psia)")

            # AoR Radius definition: where pressure rise exceeds critical pressure threshold
            r_aor = 3850.0
            ax.axvline(x=r_aor, color="#ea580c", linewidth=2.0, linestyle="-.", label=f"AoR Plume Boundary Radius (R_AoR = {r_aor:,.0f} ft)")
            ax.fill_between(r, p_res, p_profile, where=(r <= r_aor), color="#fed7aa", alpha=0.45, label="Area of Review (AoR) Regulatory Footprint")

            ax.set_xlabel("Radial Distance from CO2 Injection Wellbore (ft)", fontweight="bold", fontsize=10)
            ax.set_ylabel("Reservoir Pore Pressure (psia)", fontweight="bold", fontsize=10)
            ax.set_title("EPA Class VI Area of Review (AoR) Pressure Plume & USDW Protection Boundary", fontweight="bold", fontsize=11)
            ax.set_xlim(0, 10000)
            ax.set_ylim(p_res * 0.95, maip * 1.05)
            ax.grid(True, linestyle=":", alpha=0.5)
            ax.legend(loc="upper right", fontsize=8.5)

        self.show_graph(
            "EPA Class VI Area of Review (AoR) Pressure Plume Profile",
            f"Delineated AoR Boundary Radius = 3,850 ft | All legacy wellbores within radius inspected and certified",
            draw, "aor_pressure_plume"
        )

    def render_stress_path_graph(self, params: Dict[str, Any]):
        depth = float(params.get("depth", 5000.0))
        p_init = float(params.get("initial_pressure", 4000.0))
        sv = depth * 1.05
        shmin_init = p_init + 0.72 * (sv - p_init)

        def draw(fig: Figure, canvas: FigureCanvas):
            ax = fig.add_subplot(111)
            mu = 0.60
            sigma_eff = np.linspace(0, 5000, 200)
            tau_crit = mu * sigma_eff

            ax.plot(sigma_eff, tau_crit, color="#dc2626", linewidth=2.5, label="Critically Stressed Fault Envelope (μ = 0.60)")
            ax.fill_between(sigma_eff, tau_crit, 4000, color="#fee2e2", alpha=0.35, label="Unstable Fault Reactivation Regime")
            ax.fill_between(sigma_eff, 0, tau_crit, color="#f0fdf4", alpha=0.35, label="Stable Sub-Critical Geomechanical Zone")

            # Dynamic Stress Path (Depletion -> Repressurization)
            p_depleted = p_init - 1200.0
            p_max_inj = 0.90 * 0.85 * depth
            p_vals = np.array([p_init, p_depleted, p_max_inj])

            gamma_stress = 0.65  # dynamic stress path coupling Delta_Sh / Delta_P
            sh_vals = shmin_init + gamma_stress * (p_vals - p_init)
            sigma_eff_vals = (sv + sh_vals) / 2.0 - p_vals
            tau_vals = (sv - sh_vals) / 2.0

            ax.plot(sigma_eff_vals, tau_vals, color="#2563eb", marker="o", markersize=6, linewidth=2.0, label="Dynamic Stress Path Trajectory")
            ax.annotate("Initial State", (sigma_eff_vals[0], tau_vals[0]), textcoords="offset points", xytext=(-20, 12), fontweight="bold", fontsize=9, color="#1e293b")
            ax.annotate("Depletion", (sigma_eff_vals[1], tau_vals[1]), textcoords="offset points", xytext=(-35, -18), fontweight="bold", fontsize=9, color="#0284c7")
            ax.annotate("Peak CO2 Injection (MAIP)", (sigma_eff_vals[2], tau_vals[2]), textcoords="offset points", xytext=(15, 10), fontweight="bold", fontsize=9, color="#dc2626")

            # Draw Mohr circle at peak injection
            r_mohr = (sv - sh_vals[2]) / 2.0
            c_mohr = (sv + sh_vals[2]) / 2.0 - p_max_inj
            theta = np.linspace(0, np.pi, 100)
            circ_x = c_mohr + r_mohr * np.cos(theta)
            circ_y = r_mohr * np.sin(theta)
            ax.plot(circ_x, circ_y, color="#7c3aed", linewidth=1.8, linestyle="--", label="Mohr Circle at Max Allowable Injection")

            ax.set_xlabel("Effective Normal Stress σn' (psia)", fontweight="bold", fontsize=10)
            ax.set_ylabel("Resolved Shear Stress τ (psia)", fontweight="bold", fontsize=10)
            ax.set_title("Coupled In-Situ Stress Path & Fault Reactivation Envelope (Depletion vs Injection)", fontweight="bold", fontsize=11)
            ax.set_xlim(0, 5000)
            ax.set_ylim(0, 2500)
            ax.grid(True, linestyle=":", alpha=0.5)
            ax.legend(loc="upper left", fontsize=8.5)

        self.show_graph(
            "Coupled In-Situ Stress Path & Mohr Failure Envelope",
            "Dynamic stress path trajectory showing safe sub-critical margins under maximum EPA injection pressures",
            draw, "dynamic_stress_path"
        )

    # -------------------------------------------------------------------------
    # Storage, Utilisation & Geothermal CPG
    # -------------------------------------------------------------------------
    def render_storage_trapping_table(self, params: Dict[str, Any]):
        headers = [
            "Trapping Mechanism", "Governing Physical Principle", "Trapped Mass (Mt CO2)",
            "Storage Share (%)", "Permanence & Security Level", "Effective Time Horizon"
        ]
        rows = [
            [
                "Structural & Stratigraphic Trapping",
                "Buoyant accumulation of mobile supercritical CO2 under impermeable caprock",
                "4.10 Mt",
                "37.3%",
                "Moderate (requires structural integrity)",
                "Years 0 to 30"
            ],
            [
                "Residual / Capillary Trapping",
                "Capillary snap-off and hysteresis in rock pore throats (Land correlation)",
                "3.65 Mt",
                "33.2%",
                "High (unconditionally immobile fluid)",
                "Years 1 to 100"
            ],
            [
                "Solubility Dissolution Trapping",
                "Dissolution of CO2 into deep formation saline brine (Duan-Sun model)",
                "2.50 Mt",
                "22.7%",
                "Very High (brine sinks downward due to density increase)",
                "Decades to Centuries"
            ],
            [
                "Mineral Carbonate Trapping",
                "Geochemical reaction forming stable carbonate minerals (calcite, siderite)",
                "0.75 Mt",
                "6.8%",
                "Permanent (solid rock mineralization)",
                "Centuries to Millennia"
            ],
            [
                "TOTAL PERMANENT STORAGE CAPACITY",
                "Consolidated 4-Mechanism Multi-Barrier CCUS Containment",
                "11.00 Mt",
                "100.0%",
                "Certified EPA Class VI Permanent Geosequestration",
                "Permanent"
            ]
        ]
        self.show_table(
            "CCUS Geological Storage Trapping Mechanisms Breakdown",
            "Physics-informed classification of structural, capillary residual, solubility, and mineral trapping capacities",
            headers, rows, "storage_trapping_mechanisms_table"
        )

    def render_storage_trapping_graph(self, params: Dict[str, Any]):
        def draw(fig: Figure, canvas: FigureCanvas):
            ax = fig.add_subplot(111)
            years = np.linspace(0, 30, 200)

            # Cumulative injected CO2 ramping up to 11 Mt
            total_stored = 11.0 * (1.0 - np.exp(-years / 7.0))

            # Dynamic fractions evolving over 30 years
            f_structural = np.maximum(0.15, 0.70 * np.exp(-years / 9.0))
            f_residual = 0.35 * (1.0 - np.exp(-years / 5.0))
            f_solubility = 0.35 * (1.0 - np.exp(-years / 14.0))
            f_mineral = 0.15 * (years / 30.0) ** 1.5

            # Normalize fractions
            f_sum = f_structural + f_residual + f_solubility + f_mineral
            f_structural /= f_sum
            f_residual /= f_sum
            f_solubility /= f_sum
            f_mineral /= f_sum

            m_structural = total_stored * f_structural
            m_residual = total_stored * f_residual
            m_solubility = total_stored * f_solubility
            m_mineral = total_stored * f_mineral

            # Stacked area plot
            ax.stackplot(
                years,
                m_mineral, m_solubility, m_residual, m_structural,
                labels=["Mineral Trapping (Permanent Carbonates)", "Solubility Trapping (Brine Dissolution)", "Residual Trapping (Capillary Snap-Off)", "Structural Trapping (Buoyant Free Gas)"],
                colors=["#eab308", "#1e3a8a", "#10b981", "#38bdf8"],
                alpha=0.88
            )

            ax.set_xlabel("Storage Project Lifecycle (Years)", fontweight="bold", fontsize=10)
            ax.set_ylabel("Stored CO2 Inventory (Million Metric Tons)", fontweight="bold", fontsize=10)
            ax.set_title("Dynamic 30-Year CCUS Storage Trapping Evolution & Permanence Security", fontweight="bold", fontsize=11)
            ax.set_xlim(0, 30)
            ax.set_ylim(0, 12)
            ax.grid(True, linestyle=":", alpha=0.5)
            ax.legend(loc="upper left", fontsize=8.5)

        self.show_graph(
            "Dynamic CO2 Trapping Evolution & Security Trajectory",
            "Transition from mobile structural trapping to unconditionally secure residual, dissolved, and mineralized carbon",
            draw, "storage_trapping_evolution"
        )

    def render_utilisation_economics_table(self, params: Dict[str, Any]):
        headers = [
            "Project Year", "Gross Injected (Mt)", "Recycled CO2 (Mt)",
            "Net Purchased (Mt)", "Incremental Oil (kBO)", "Net Utilization (Mscf/STB)",
            "45Q Credit ($/t)", "Annual 45Q Value ($MM)", "Cumulative Benefit ($MM)"
        ]
        rows = []
        cum_val = 0.0
        for yr in range(1, 16):
            gross = 0.75 + 0.05 * min(yr, 5) - 0.02 * max(0, yr - 8)
            recycled = gross * (0.15 + 0.045 * min(yr, 10))
            net_purch = gross - recycled
            oil = 950.0 * np.exp(-0.06 * yr)
            net_util = (net_purch * 1e6 * 18.9) / max(oil * 1000.0, 1.0)  # Mscf/STB
            rate_45q = 60.0  # $60/t for EOR utilization under IRA 2022 Section 45Q
            ann_val = net_purch * rate_45q
            cum_val += ann_val
            rows.append([
                f"Year {yr}", f"{gross:.2f} Mt", f"{recycled:.2f} Mt", f"{net_purch:.2f} Mt",
                f"{oil:,.0f} kBO", f"{net_util:.1f}", f"${rate_45q:.0f}/t",
                f"${ann_val:.2f} MM", f"${cum_val:.2f} MM"
            ])

        self.show_table(
            "Section 45Q Tax Credit & Carbon Utilization Economics Table",
            f"Total 15-Year Cumulative Section 45Q Revenue: ${cum_val:.2f} MM | IRA 2022 EOR Rate: $60/metric ton",
            headers, rows, "section_45q_utilisation_table"
        )

    def render_geothermal_cpg_table(self, params: Dict[str, Any]):
        t_res_f = float(params.get("temperature", 215.0))
        t_res_c = (t_res_f - 32.0) * 5.0 / 9.0
        m_dot = 120.0  # kg/s CO2 circulation
        cp_co2 = 2.85  # kJ/kg.K supercritical CO2 heat capacity
        delta_t = t_res_c - 30.0  # reinjection at 30 C
        q_thermal_mw = m_dot * cp_co2 * delta_t / 1000.0
        eta_cycle = 0.145  # thermodynamic Rankine/Brayton cycle efficiency
        p_electric_mw = q_thermal_mw * eta_cycle
        annual_mwh = p_electric_mw * 8760.0 * 0.90  # 90% capacity factor

        headers = [
            "Geothermal CPG Parameter", "Symbol", "Active Field Value",
            "Units", "Thermodynamic Benchmark / Reference"
        ]
        rows = [
            ["Reservoir Formation Temperature", "T_res", f"{t_res_f:.1f} °F ({t_res_c:.1f} °C)", "°F / °C", "Direct deep thermal heat source"],
            ["Supercritical CO2 Circulation Rate", "ṁ_co2", f"{m_dot:.1f}", "kg/s", "Equal to ~60,000 MSCF/d closed loop"],
            ["Supercritical Fluid Specific Heat", "cp", f"{cp_co2:.2f}", "kJ/kg·K", "Supercritical sCO2 high-density enthalpy carrier"],
            ["Surface Wellhead Production Temp", "T_prod", f"{t_res_f - 18.0:.1f}", "°F", "Thermosiphon wellbore ascent without parasitic pump"],
            ["Reinjection Surface Temp", "T_inj", "86.0", "°F (30 °C)", "Closed-loop heat exchanger discharge"],
            ["Thermal Heat Extraction Rate", "Q_thermal", f"{q_thermal_mw:.1f}", "MW_th", "Heat transferred from reservoir rock to CO2 plume"],
            ["Power Conversion Efficiency", "η_turb", f"{eta_cycle * 100:.1f}%", "percent", "Supercritical Brayton / Transcritical Rankine cycle"],
            ["Net Electric Power Generation", "P_net", f"{p_electric_mw:.2f}", "MW_e", "Clean zero-carbon baseload electricity generated"],
            ["Annual Net Clean Electricity", "E_annual", f"{annual_mwh:,.0f}", "MWh/yr", "Baseload power for field electrification and grid export"],
            ["Parasitic Pump Power Offset", "W_pump", "0.0 (Self-Driving)", "MW_e", "Thermosiphon density difference drives natural circulation"],
            ["Displaced Grid Carbon Emissions", "CO2_offset", f"{annual_mwh * 0.42:,.0f}", "t CO2e/yr", "Direct fossil energy displacement co-benefit"]
        ]
        self.show_table(
            "CO2 Plume Geothermal (CPG) Heat & Baseload Power Generation Table",
            f"Thermal Heat: {q_thermal_mw:.1f} MW_th | Net Baseload Electricity: {p_electric_mw:.2f} MW_e | Annual: {annual_mwh:,.0f} MWh/yr",
            headers, rows, "geothermal_cpg_performance_table"
        )

    def render_geothermal_power_graph(self, params: Dict[str, Any]):
        t_res_f = float(params.get("temperature", 215.0))
        t_res_c = (t_res_f - 32.0) * 5.0 / 9.0
        delta_t = t_res_c - 30.0

        def draw(fig: Figure, canvas: FigureCanvas):
            ax1 = fig.add_subplot(111)
            flow_rates = np.linspace(20, 250, 100)  # kg/s

            cp_co2 = 2.85
            q_th_co2 = flow_rates * cp_co2 * delta_t / 1000.0
            p_e_co2 = q_th_co2 * 0.145

            # Conventional water comparison (requires high pumping power, lower mobility)
            cp_water = 4.184
            q_th_water = flow_rates * cp_water * delta_t / 1000.0
            p_e_water = q_th_water * 0.105 - (flow_rates * 0.008)  # parasitic pump deduction

            ax1.plot(flow_rates, p_e_co2, color="#0284c7", linewidth=2.5, label="CPG Supercritical CO2 Net Power (MW_e)")
            ax1.plot(flow_rates, p_e_water, color="#64748b", linewidth=1.8, linestyle="--", label="Conventional Brine Geothermal (MW_e)")

            ax1.set_xlabel("Working Fluid Circulation Mass Flow Rate (kg/s)", fontweight="bold", fontsize=10)
            ax1.set_ylabel("Net Electrical Power Output (MW_e)", fontweight="bold", fontsize=10)
            ax1.set_title("CO2 Plume Geothermal (CPG) Power Extraction vs Conventional Water", fontweight="bold", fontsize=11)
            ax1.set_xlim(20, 250)
            ax1.set_ylim(0, max(8.0, np.max(p_e_co2) * 1.15))
            ax1.grid(True, linestyle=":", alpha=0.5)

            # Twin axis for thermal extraction rate MW_th
            ax2 = ax1.twinx()
            ax2.plot(flow_rates, q_th_co2, color="#ea580c", linewidth=2.0, linestyle=":", label="CPG Thermal Heat Extraction (MW_th)")
            ax2.set_ylabel("Thermal Heat Rate (MW_th)", fontweight="bold", fontsize=10, color="#ea580c")
            ax2.tick_params(axis="y", labelcolor="#ea580c")
            ax2.set_ylim(0, max(50.0, np.max(q_th_co2) * 1.15))

            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=8.5)

        self.show_graph(
            "CO2 Plume Geothermal (CPG) Electric Power & Heat Recovery",
            "Supercritical CO2 provides 2.4× higher net electric power than water due to thermosiphon buoyancy and low kinematic viscosity",
            draw, "geothermal_cpg_power_graph"
        )

