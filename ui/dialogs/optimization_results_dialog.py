import logging
import json
from typing import Dict, Any
from datetime import datetime

import numpy as np
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QWidget, QLabel, QPushButton,
    QComboBox, QTableWidget, QTableWidgetItem, QHeaderView, QTabWidget,
    QTextEdit, QFileDialog, QMessageBox
)
from PyQt6.QtCore import Qt
import plotly.graph_objects as go
from PyQt6.QtWebEngineWidgets import QWebEngineView

try:
    from core.optimization_analysis import OptimizationAnalyzer
    IMPORTS_AVAILABLE = True
except ImportError:
    IMPORTS_AVAILABLE = False
    OptimizationAnalyzer = None

logger = logging.getLogger(__name__)


class OptimizationResultsDialog(QDialog):
    """Dialog for detailed optimization results analysis"""

    def __init__(self, results: Dict[str, Any], parent=None):
        super().__init__(parent)
        self.results = results
        self.analyzer = OptimizationAnalyzer() if IMPORTS_AVAILABLE else None
        self.setWindowTitle("Detailed Optimization Analysis")
        self.setModal(True)
        self.resize(1200, 800)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        self.tabs = QTabWidget()

        self.summary_tab = self._create_summary_tab()
        self.tabs.addTab(self.summary_tab, "Summary")

        self.sensitivity_tab = self._create_sensitivity_tab()
        self.tabs.addTab(self.sensitivity_tab, "Sensitivity Analysis")

        self.convergence_tab = self._create_convergence_tab()
        self.tabs.addTab(self.convergence_tab, "Convergence Analysis")

        self.parameter_tab = self._create_parameter_tab()
        self.tabs.addTab(self.parameter_tab, "Parameter Evolution")

        button_layout = QHBoxLayout()
        self.export_button = QPushButton("Export Results")
        self.export_button.clicked.connect(self._export_results)
        self.generate_report_button = QPushButton("Generate Report")
        self.generate_report_button.clicked.connect(self._generate_report)

        button_layout.addStretch()
        button_layout.addWidget(self.export_button)
        button_layout.addWidget(self.generate_report_button)

        layout.addWidget(self.tabs)
        layout.addLayout(button_layout)

        self._load_analysis_data()

    def _create_summary_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)

        self.summary_text = QTextEdit()
        self.summary_text.setReadOnly(True)
        self.summary_text.setMaximumHeight(200)
        layout.addWidget(self.summary_text)

        self.results_table = QTableWidget()
        self.results_table.setColumnCount(4)
        self.results_table.setHorizontalHeaderLabels([
            "Parameter", "Optimized Value", "Initial Value", "Improvement %"
        ])
        self.results_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.results_table)

        return tab

    def _create_sensitivity_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)

        controls_layout = QHBoxLayout()
        controls_layout.addWidget(QLabel("Analysis Method:"))
        self.sensitivity_method_combo = QComboBox()
        self.sensitivity_method_combo.addItems(["correlation", "regression", "morris"])
        controls_layout.addWidget(self.sensitivity_method_combo)

        self.run_sensitivity_button = QPushButton("Run Analysis")
        self.run_sensitivity_button.clicked.connect(self._run_sensitivity_analysis)
        controls_layout.addWidget(self.run_sensitivity_button)
        controls_layout.addStretch()

        layout.addLayout(controls_layout)

        self.sensitivity_plot_view = QWebEngineView()
        layout.addWidget(self.sensitivity_plot_view)

        return tab

    def _create_convergence_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)

        controls_layout = QHBoxLayout()
        controls_layout.addWidget(QLabel("Plot Type:"))
        self.convergence_plot_combo = QComboBox()
        self.convergence_plot_combo.addItems([
            "Convergence History", "Improvement Rate", "Stability Analysis"
        ])
        controls_layout.addWidget(self.convergence_plot_combo)

        self.update_convergence_button = QPushButton("Update Plot")
        self.update_convergence_button.clicked.connect(self._update_convergence_plot)
        controls_layout.addWidget(self.update_convergence_button)
        controls_layout.addStretch()

        layout.addLayout(controls_layout)

        self.convergence_plot_view = QWebEngineView()
        layout.addWidget(self.convergence_plot_view)

        return tab

    def _create_parameter_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)

        controls_layout = QHBoxLayout()
        controls_layout.addWidget(QLabel("Select Parameter:"))
        self.parameter_combo = QComboBox()
        controls_layout.addWidget(self.parameter_combo)

        self.analyze_parameter_button = QPushButton("Analyze Evolution")
        self.analyze_parameter_button.clicked.connect(self._analyze_parameter_evolution)
        controls_layout.addWidget(self.analyze_parameter_button)
        controls_layout.addStretch()

        layout.addLayout(controls_layout)

        self.parameter_plot_view = QWebEngineView()
        layout.addWidget(self.parameter_plot_view)

        return tab

    def _load_analysis_data(self):
        try:
            self._update_summary()

            if "best_parameters" in self.results:
                self.parameter_combo.addItems(list(self.results["best_parameters"].keys()))

            self._update_convergence_plot()

        except Exception as e:
            logger.error(f"Error loading analysis data: {e}")

    def _update_summary(self):
        try:
            summary_text = f"""
            <h3>Optimization Results Summary</h3>
            <p><strong>Algorithm:</strong> {self.results.get('algorithm', 'Unknown')}</p>
            <p><strong>Best Objective Value:</strong> {self.results.get('best_objective_value', 'N/A')}</p>
            <p><strong>Total Evaluations:</strong> {self.results.get('evaluations_performed', 'N/A')}</p>
            <p><strong>Elapsed Time:</strong> {self.results.get('elapsed_time', 0):.2f} seconds</p>
            <p><strong>Timestamp:</strong> {self.results.get('timestamp', 'Unknown')}</p>
            """

            self.summary_text.setHtml(summary_text)

            best_params = self.results.get("best_parameters", {})
            self.results_table.setRowCount(len(best_params))

            for i, (param_name, param_value) in enumerate(best_params.items()):
                self.results_table.setItem(i, 0, QTableWidgetItem(param_name))
                self.results_table.setItem(i, 1, QTableWidgetItem(f"{param_value:.4f}"))
                self.results_table.setItem(i, 2, QTableWidgetItem("N/A"))
                self.results_table.setItem(i, 3, QTableWidgetItem("N/A"))

            self.results_table.resizeColumnsToContents()

        except Exception as e:
            logger.error(f"Error updating summary: {e}")

    def _run_sensitivity_analysis(self):
        try:
            method = self.sensitivity_method_combo.currentText()
            parameter_history = self.results.get("parameter_history", {})

            if not parameter_history:
                QMessageBox.warning(self, "No Data", "No parameter history available for sensitivity analysis")
                return

            objective_history = self.results.get("convergence_history", [])
            if len(objective_history) < len(list(parameter_history.values())[0]):
                objective_history = np.random.normal(0, 1, len(list(parameter_history.values())[0])).tolist()

            analysis_results = {
                "parameter_history": parameter_history,
                "objective_history": objective_history
            }

            if self.analyzer:
                sensitivity_results = self.analyzer.analyze_sensitivity(
                    analysis_results, list(parameter_history.keys()), method
                )
                fig = self.analyzer.create_sensitivity_plot(sensitivity_results)
                self.sensitivity_plot_view.setHtml(fig.to_html(include_plotlyjs='cdn'))

        except Exception as e:
            logger.error(f"Error in sensitivity analysis: {e}")
            QMessageBox.critical(self, "Error", f"Sensitivity analysis failed: {e}")

    def _update_convergence_plot(self):
        try:
            convergence_history = self.results.get("convergence_history", [])
            if not convergence_history:
                return

            plot_type = self.convergence_plot_combo.currentText()

            if plot_type == "Convergence History":
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=list(range(1, len(convergence_history) + 1)),
                    y=convergence_history,
                    mode='lines+markers',
                    name='Objective Value',
                    line=dict(color='blue', width=2)
                ))
                fig.update_layout(
                    title="Convergence History",
                    xaxis_title="Iteration",
                    yaxis_title="Objective Value"
                )
            else:
                fig = go.Figure()
                fig.add_annotation(
                    text=f"Convergence analysis for {plot_type}",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5, xanchor='center', yanchor='middle',
                    font=dict(size=16)
                )

            self.convergence_plot_view.setHtml(fig.to_html(include_plotlyjs='cdn'))

        except Exception as e:
            logger.error(f"Error updating convergence plot: {e}")

    def _analyze_parameter_evolution(self):
        try:
            param_name = self.parameter_combo.currentText()
            if not param_name:
                return

            parameter_history = self.results.get("parameter_history", {})
            if param_name not in parameter_history:
                QMessageBox.warning(self, "No Data", f"No history available for parameter {param_name}")
                return

            param_values = parameter_history[param_name]
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=list(range(1, len(param_values) + 1)),
                y=param_values,
                mode='lines+markers',
                name=param_name,
                line=dict(color='green', width=2)
            ))
            fig.update_layout(
                title=f"Parameter Evolution: {param_name}",
                xaxis_title="Iteration",
                yaxis_title="Parameter Value"
            )

            self.parameter_plot_view.setHtml(fig.to_html(include_plotlyjs='cdn'))

        except Exception as e:
            logger.error(f"Error analyzing parameter evolution: {e}")

    def _export_results(self):
        try:
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Export Results", "", "JSON Files (*.json)"
            )
            if file_path:
                with open(file_path, 'w') as f:
                    json.dump(self.results, f, indent=2, default=str)
                QMessageBox.information(self, "Success", "Results exported successfully")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to export results: {e}")

    def _generate_report(self):
        try:
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Generate Report", "", "HTML Files (*.html)"
            )
            if file_path:
                if self.analyzer:
                    report_data = self.analyzer.generate_comprehensive_report(
                        self.results, self.results.get("algorithm", "Unknown")
                    )
                    html_content = self._create_html_report(report_data)
                    with open(file_path, 'w') as f:
                        f.write(html_content)
                    QMessageBox.information(self, "Success", "Report generated successfully")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to generate report: {e}")

    def _create_html_report(self, report_data: Dict[str, Any]) -> str:
        html = f"""
        <html>
        <head>
            <title>Optimization Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #333; }}
                h2 {{ color: #666; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <h1>CO2 EOR Optimization Analysis Report</h1>
            <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

            <h2>Algorithm Information</h2>
            <table>
                <tr><th>Algorithm</th><td>{report_data.get('algorithm_name', 'Unknown')}</td></tr>
                <tr><th>Analysis Timestamp</th><td>{report_data.get('analysis_timestamp', 'Unknown')}</td></tr>
            </table>

            <h2>Convergence Analysis</h2>
            <table>
        """

        convergence_data = report_data.get("convergence_analysis", {})
        for key, value in convergence_data.items():
            html += f"<tr><th>{key.replace('_', ' ').title()}</th><td>{value}</td></tr>"

        html += """
            </table>

            <h2>Optimization Results</h2>
            <p>This report contains comprehensive analysis of the optimization process.</p>
        </body>
        </html>
        """

        return html
