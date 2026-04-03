import os
import pdfkit
from PyQt6.QtCore import QObject, pyqtSignal
import datetime
from typing import Dict, List, Any, Optional
import numpy as np
import json
import base64
import io
import plotly.graph_objects as go
from plotly.io import to_image
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
import logging

from utils.units_manager import UnitsManager
from core.data_models import (
    ReservoirData, PVTProperties, EORParameters, EconomicParameters,
    OperationalParameters, ProfileParameters, EOSModelParameters, WellData
)
from analysis.decline_curve_analysis import DeclineCurveAnalyzer, DCAResult

class ReportGenerator(QObject):
    progress_updated = pyqtSignal(int, str)
    report_generated = pyqtSignal(str)
    error_occurred = pyqtSignal(str)

    def __init__(self, units_manager: UnitsManager):
        super().__init__()
        self.units_manager = units_manager
        self.wkhtml_path = self.find_wkhtmltopdf()

    def find_wkhtmltopdf(self):
        try:
            import shutil
            path = shutil.which("wkhtmltopdf")
            if path and os.path.exists(path):
                self._log_diagnostic_info(f"Found wkhtmltopdf in PATH: {path}")
                return path

            common_paths = [
                "wkhtmltopdf/bin/wkhtmltopdf.exe",
                "C:/Program Files/wkhtmltopdf/bin/wkhtmltopdf.exe",
                "C:/Program Files (x86)/wkhtmltopdf/bin/wkhtmltopdf.exe",
                os.path.expanduser("~/AppData/Local/Programs/wkhtmltopdf/bin/wkhtmltopdf.exe"),
                "C:/wkhtmltopdf/bin/wkhtmltopdf.exe",
                "D:/Program Files/wkhtmltopdf/bin/wkhtmltopdf.exe",
                "D:/Program Files/wkhtmltopdf/wkhtmltopdf.exe"
            ]
            for path in common_paths:
                if os.path.exists(path):
                    abs_path = os.path.abspath(path)
                    self._log_diagnostic_info(f"Found wkhtmltopdf at: {abs_path}")
                    return abs_path

            self._log_diagnostic_info("wkhtmltopdf not found in any standard locations")
            self._log_diagnostic_info("Please install the actual wkhtmltopdf binary from https://wkhtmltopdf.org/")
            self._log_diagnostic_info("The Python 'wkhtmltopdf' package is just a wrapper and doesn't include the binary.")
            return None
        except Exception as e:
            self._log_diagnostic_info(f"Error finding wkhtmltopdf: {str(e)}")
            return None

    def _log_diagnostic_info(self, message):
        logging.info(f"ReportGenerator Diagnostic: {message}")
        self.progress_updated.emit(0, message)

    def test_wkhtmltopdf_installation(self):
        try:
            if self.wkhtml_path is None:
                return "wkhtmltopdf binary not found. Please install from https://wkhtmltopdf.org/"
            import subprocess
            result = subprocess.run([self.wkhtml_path, '--version'],
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                return f"wkhtmltopdf is working: {result.stdout.strip()}"
            else:
                return f"wkhtmltopdf test failed: {result.stderr}"
        except Exception as e:
            return f"Error testing wkhtmltopdf: {str(e)}"

    def generate_report(self, report_data: Dict[str, Any], output_path: str, config: Dict[str, Any]) -> bool:
        """Generate a complete report with all selected sections"""
        try:
            self.progress_updated.emit(0, "Starting report generation")
            html_content = self._generate_html_content(report_data, config)
            if self.wkhtml_path is None:
                self.progress_updated.emit(0, "wkhtmltopdf not found. Saving as HTML instead.")
                html_output_path = output_path.replace('.pdf', '.html')
                with open(html_output_path, 'w', encoding='utf-8') as f:
                    f.write(html_content)
                self.progress_updated.emit(100, f"Report saved as HTML: {html_output_path}")
                self.report_generated.emit(html_output_path)
                return True
            
            self.progress_updated.emit(30, "Converting HTML to PDF")
            # Get margin settings from config, default to 20mm
            format_options = config.get('format_options', {})
            margins = format_options.get('margins', {})
            margin_top = margins.get('top', 20)
            margin_right = margins.get('right', 20)
            margin_bottom = margins.get('bottom', 20)
            margin_left = margins.get('left', 20)
            
            options = {
                'enable-local-file-access': None,
                'quiet': '',
                'margin-top': f'{margin_top}mm',
                'margin-right': f'{margin_right}mm',
                'margin-bottom': f'{margin_bottom}mm',
                'margin-left': f'{margin_left}mm',
                'encoding': "UTF-8",
            }

            config_pdfkit = pdfkit.configuration(wkhtmltopdf=self.wkhtml_path)
            
            pdfkit.from_string(
                html_content,
                output_path,
                options=options,
                configuration=config_pdfkit
            )
                
            self.progress_updated.emit(100, "Report generated successfully")
            self.report_generated.emit(output_path)
            return True
        except Exception as e:
            error_msg = f"PDF generation error: {str(e)}"
            self.progress_updated.emit(0, error_msg)
            try:
                html_output_path = output_path.replace('.pdf', '.html')
                with open(html_output_path, 'w', encoding='utf-8') as f:
                    f.write(html_content)
                self.error_occurred.emit(f"PDF generation failed: {str(e)}. Report saved as HTML: {html_output_path}")
            except Exception as fallback_error:
                self.error_occurred.emit(f"PDF generation failed: {str(e)} and HTML fallback also failed: {str(fallback_error)}")
            return False

    def _generate_html_content(self, report_data: Dict[str, Any], config: Dict[str, Any]) -> str:
        sections = []

        if config.get('sections', {}).get('executive_summary', True):
            sections.append(self._generate_cover_page(report_data))

        sections.append(self._generate_table_of_contents(config))

        if config.get('sections', {}).get('executive_summary', True):
            sections.append(self._generate_executive_summary(report_data))

        if config.get('sections', {}).get('input_parameters', True):
            sections.append(self._generate_input_parameters(report_data))

        if config.get('sections', {}).get('optimization_results', True) and 'optimization_results' in report_data:
            sections.append(self._generate_optimization_results(report_data, config))

        if config.get('sections', {}).get('sensitivity_analysis', True) and 'sensitivity_results' in report_data:
            sections.append(self._generate_sensitivity_analysis(report_data, config))

        if config.get('sections', {}).get('uq_analysis', True) and 'uq_results' in report_data:
            sections.append(self._generate_uncertainty_quantification(report_data, config))

        if config.get('sections', {}).get('economic_assumptions', True):
            sections.append(self._generate_economic_analysis(report_data, config))

        if config.get('sections', {}).get('decline_curve_analysis', True) and 'dca_results' in report_data:
            sections.append(self._generate_decline_curve_analysis(report_data, config))

        if config.get('sections', {}).get('decline_curve_analysis', True) and 'dca_results' in report_data:
            sections.append(self._generate_decline_curve_analysis(report_data, config))

        if config.get('sections', {}).get('validation_report', True) and 'validation_report' in report_data:
            sections.append(self._generate_validation_report(report_data))

        if config.get('sections', {}).get('appendices', True):
            sections.append(self._generate_appendices(report_data))

        css_content = self._get_css_styles(config)

        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>CO2 EOR Optimization Report</title>
            <style>
            {css_content}
            </style>
        </head>
        <body>
            <div class="report-container">
                {"".join(sections)}
            </div>
        </body>
        </html>
        """
        
        return html_content

    def _generate_cover_page(self, report_data: Dict[str, Any]) -> str:
        current_date = datetime.datetime.now().strftime("%Y-%m-%d")
        session_id = report_data.get('session_id', 'N/A')
        return f"""
        <div class="cover-page">
            <h1>CO2 Enhanced Oil Recovery Optimization Report</h1>
            <h2>Comprehensive Technical and Economic Analysis</h2>
            <div class="cover-details">
                <p><strong>Date:</strong> {current_date}</p>
                <p><strong>Project:</strong> {report_data.get('project_name', 'Unnamed Project')}</p>
                <p><strong>Session ID:</strong> {session_id}</p>
                <p><strong>Generated by:</strong> CO2EOR Optimizer Software</p>
            </div>
        </div>
        <div class="page-break"></div>
        """

    def _generate_table_of_contents(self, config: Dict[str, Any]) -> str:
        toc_items = []
        sections = config.get('sections', {})
        
        if sections.get('executive_summary', True):
            toc_items.append('<li><a href="#executive-summary">Executive Summary</a></li>')
        if sections.get('input_parameters', True):
            toc_items.append('<li><a href="#input-parameters">Input Parameters</a></li>')
        if sections.get('optimization_results', True):
            toc_items.append('<li><a href="#optimization-results">Optimization Results</a></li>')
        if sections.get('sensitivity_analysis', True):
            toc_items.append('<li><a href="#sensitivity-analysis">Sensitivity Analysis</a></li>')
        if sections.get('uq_analysis', True):
            toc_items.append('<li><a href="#uq-analysis">Uncertainty Quantification</a></li>')
        if sections.get('economic_assumptions', True):
            toc_items.append('<li><a href="#economic-assumptions">Economic Assumptions</a></li>')
        if sections.get('decline_curve_analysis', True):
            toc_items.append('<li><a href="#decline-curve-analysis">Decline Curve Analysis</a></li>')
        if sections.get('validation_report', True):
            toc_items.append('<li><a href="#validation-report">Model Validation</a></li>')
        if sections.get('appendices', True):
            toc_items.append('<li><a href="#appendices">Appendices</a></li>')

        return f"""
        <div class="table-of-contents">
            <h2>Table of Contents</h2>
            <ul>
                {"".join(toc_items)}
            </ul>
        </div>
        <div class="page-break"></div>
        """

    def _generate_executive_summary(self, report_data: Dict[str, Any]) -> str:
        """Generate executive summary HTML"""
        optimization_results = report_data.get('optimization_results', {})
        economic_data = report_data.get('economic_parameters', {})
        
        best_npv = optimization_results.get('final_metrics', {}).get('npv', 0)
        recovery_factor = optimization_results.get('final_metrics', {}).get('recovery_factor', 0)
        co2_utilization = optimization_results.get('final_metrics', {}).get('co2_utilization', 0)
        
        return f"""
        <div id="executive-summary" class="section">
            <h2>Executive Summary</h2>
            <div class="key-findings">
                <h3>Key Findings</h3>
                <div class="summary-metrics">
                    <div class="metric">
                        <span class="metric-value">${best_npv:,.0f}</span>
                        <span class="metric-label">Net Present Value</span>
                    </div>
                    <div class="metric">
                        <span class="metric-value">{recovery_factor:.1f}%</span>
                        <span class="metric-label">Recovery Factor</span>
                    </div>
                    <div class="metric">
                        <span class="metric-value">{co2_utilization:.2f}</span>
                        <span class="metric-label">CO2 Utilization (bbl/ton)</span>
                    </div>
                </div>
                
                <div class="recommendations">
                    <h3>Recommendations</h3>
                    <ul>
                        <li>Implement optimized injection strategy with continuous monitoring</li>
                        <li>Consider phased development approach based on sensitivity analysis</li>
                        <li>Monitor reservoir response and adjust parameters as needed</li>
                        <li>Evaluate economic viability under different price scenarios</li>
                    </ul>
                </div>
            </div>
        </div>
        <div class="page-break"></div>
        """
    def _generate_input_parameters(self, report_data: Dict[str, Any]) -> str:
        """Generate comprehensive input parameters section HTML"""
        reservoir_data = report_data.get('reservoir_data', {})
        pvt_data = report_data.get('pvt_data', {})
        well_data = report_data.get('well_data', [])
        eor_params = report_data.get('eor_parameters', {})
        operational_params = report_data.get('operational_parameters', {})
        profile_params = report_data.get('profile_parameters', {})

        grid_dims = reservoir_data.get('runspec', {}).get('DIMENSIONS', 'N/A')
        ooip = reservoir_data.get('ooip_stb', 0)

        poro_array = reservoir_data.get('grid', {}).get('PORO')
        if poro_array is not None and hasattr(poro_array, '__len__') and len(poro_array) > 0:
            porosity = np.mean(poro_array)
        else:
            porosity = 0

        permx_array = reservoir_data.get('grid', {}).get('PERMX')
        if permx_array is not None and hasattr(permx_array, '__len__') and len(permx_array) > 0:
            permeability = np.mean(permx_array)
        else:
            permeability = 0
        
        return f"""
        <div id="input-parameters" class="section">
            <h2>Input Parameters</h2>
            
            <div class="subsection">
                <h3>Reservoir Characteristics</h3>
                <table class="data-table">
                    <tr><th>Parameter</th><th>Value</th><th>Units</th></tr>
                    <tr><td>Grid Dimensions</td><td>{grid_dims}</td><td>-</td></tr>
                    <tr><td>OOIP</td><td>{ooip:,.0f}</td><td>STB</td></tr>
                    <tr><td>Average Porosity</td><td>{porosity:.3f}</td><td>v/v</td></tr>
                    <tr><td>Average Permeability</td><td>{permeability:.2f}</td><td>md</td></tr>
                </table>
            </div>
            
            <div class="subsection">
                <h3>Fluid Properties</h3>
                <table class="data-table">
                    <tr><th>Parameter</th><th>Value</th><th>Units</th></tr>
                    <tr><td>Oil Gravity</td><td>{pvt_data.get('gas_specific_gravity', 'N/A')}</td><td>API</td></tr>
                    <tr><td>Reservoir Temperature</td><td>{pvt_data.get('temperature', 'N/A')}</td><td>°F</td></tr>
                    <tr><td>PVT Type</td><td>{pvt_data.get('pvt_type', 'N/A')}</td><td>-</td></tr>
                </table>
            </div>
            
            <div class="subsection">
                <h3>Well Data Summary</h3>
                <table class="data-table">
                    <tr><th>Well Name</th><th>Depth Range (ft)</th><th>Properties Measured</th></tr>
                    {"".join([f'<tr><td>{well.get("name", "N/A")}</td><td>{np.min(well.get("depths", [0])):.0f} - {np.max(well.get("depths", [0])):.0f}</td><td>{", ".join(well.get("properties", {}).keys())}</td></tr>' for well in well_data])}
                </table>
            </div>
            
            <div class="subsection">
                <h3>EOR Parameters</h3>
                <table class="data-table">
                    <tr><th>Parameter</th><th>Value</th><th>Units</th></tr>
                    <tr><td>Injection Rate</td><td>{eor_params.get('injection_rate', 'N/A')}</td><td>bpd</td></tr>
                    <tr><td>WAG Ratio</td><td>{eor_params.get('WAG_ratio', 'N/A')}</td><td>-</td></tr>
                    <tr><td>Injection Scheme</td><td>{eor_params.get('injection_scheme', 'N/A')}</td><td>-</td></tr>
                    <tr><td>Target Pressure</td><td>{eor_params.get('target_pressure_psi', 'N/A')}</td><td>psi</td></tr>
                    <tr><td>Max Pressure</td><td>{eor_params.get('max_pressure_psi', 'N/A')}</td><td>psi</td></tr>
                    <tr><td>Min Injection Rate</td><td>{eor_params.get('min_injection_rate_bpd', 'N/A')}</td><td>bpd</td></tr>
                    <tr><td>Max Injection Rate</td><td>{eor_params.get('max_injection_rate_bpd', 'N/A')}</td><td>bpd</td></tr>
                    <tr><td>VDP Coefficient</td><td>{eor_params.get('v_dp_coefficient', 'N/A')}</td><td>-</td></tr>
                    <tr><td>Mobility Ratio</td><td>{eor_params.get('mobility_ratio', 'N/A')}</td><td>-</td></tr>
                </table>
            </div>
            
            <div class="subsection">
                <h3>Operational Parameters</h3>
                <table class="data-table">
                    <tr><th>Parameter</th><th>Value</th><th>Units</th></tr>
                    <tr><td>Project Lifetime</td><td>{operational_params.get('project_lifetime_years', 'N/A')}</td><td>years</td></tr>
                    <tr><td>Target Objective</td><td>{operational_params.get('target_objective_name', 'N/A')}</td><td>-</td></tr>
                    <tr><td>Target Value</td><td>{operational_params.get('target_objective_value', 'N/A')}</td><td>-</td></tr>
                    <tr><td>Recovery Model</td><td>{report_data.get('recovery_model', 'N/A')}</td><td>-</td></tr>
                </table>
            </div>
            
            <div class="subsection">
                <h3>Profile Parameters</h3>
                <table class="data-table">
                    <tr><th>Parameter</th><th>Value</th><th>Units</th></tr>
                    <tr><td>Oil Profile Type</td><td>{profile_params.get('oil_profile_type', 'N/A')}</td><td>-</td></tr>
                    <tr><td>Injection Profile Type</td><td>{profile_params.get('injection_profile_type', 'N/A')}</td><td>-</td></tr>
                    <tr><td>Plateau Duration</td><td>{profile_params.get('plateau_duration_fraction_of_life', 'N/A')}</td><td>fraction</td></tr>
                    <tr><td>Initial Decline Rate</td><td>{profile_params.get('initial_decline_rate_annual_fraction', 'N/A')}</td><td>fraction/year</td></tr>
                    <tr><td>CO2 Breakthrough</td><td>{profile_params.get('co2_breakthrough_year_fraction', 'N/A')}</td><td>fraction</td></tr>
                    <tr><td>CO2 Production Ratio</td><td>{profile_params.get('co2_production_ratio_after_breakthrough', 'N/A')}</td><td>-</td></tr>
                    <tr><td>CO2 Recycling Efficiency</td><td>{profile_params.get('co2_recycling_efficiency_fraction', 'N/A')}</td><td>fraction</td></tr>
                </table>
            </div>
        </div>
        <div class="page-break"></div>
        """

    def _generate_optimization_results(self, report_data: Dict[str, Any], config: Dict[str, Any]) -> str:
        """Generate optimization results HTML"""
        results = report_data.get('optimization_results', {})
        optimized_params = results.get('optimized_params_final_clipped', {})
        final_metrics = results.get('final_metrics', {})

        charts_html = ""
        if config.get('format_options', {}).get('include_charts', True) and 'charts' in report_data:
            charts = report_data['charts']
            if 'optimization_convergence' in charts:
                charts_html += f"""
                <div class="chart-container">
                    <h4>Optimization Convergence</h4>
                    <img src="{charts['optimization_convergence']}" alt="Optimization Convergence Chart" class="chart-image">
                </div>
                """
            if 'production_profiles' in charts:
                charts_html += f"""
                <div class="chart-container">
                    <h4>Production Profiles</h4>
                    <img src="{charts['production_profiles']}" alt="Production Profiles Chart" class="chart-image">
                </div>
                """
            if 'ga_coverage_distribution' in charts:
                charts_html += f"""
                <div class="chart-container">
                    <h4>GA Coverage Distribution</h4>
                    <img src="{charts['ga_coverage_distribution']}" alt="GA Coverage Distribution Chart" class="chart-image">
                </div>
                """
            if 'ga_objective_distribution' in charts:
                charts_html += f"""
                <div class="chart-container">
                    <h4>GA Objective Distribution</h4>
                    <img src="{charts['ga_objective_distribution']}" alt="GA Objective Distribution Chart" class="chart-image">
                </div>
                """
            if 'hybrid_model_analysis' in charts:
                charts_html += f"""
                <div class="chart-container">
                    <h4>Hybrid Model Analysis</h4>
                    <img src="{charts['hybrid_model_analysis']}" alt="Hybrid Model Analysis Chart" class="chart-image">
                </div>
                """
            if 'breakthrough_mechanism_analysis' in charts:
                charts_html += f"""
                <div class="chart-container">
                    <h4>Breakthrough Mechanism Analysis</h4>
                    <img src="{charts['breakthrough_mechanism_analysis']}" alt="Breakthrough Mechanism Analysis Chart" class="chart-image">
                </div>
                """
        
        return f"""
        <div id="optimization-results" class="section">
            <h2>Optimization Results</h2>
            
            <div class="subsection">
                <h3>Optimized Parameters</h3>
                <table class="data-table">
                    <tr><th>Parameter</th><th>Optimized Value</th><th>Units</th></tr>
                    {"".join([f'<tr><td>{param}</td><td>{value:.4f}</td><td>-</td></tr>' for param, value in optimized_params.items()])}
                </table>
            </div>
            
            <div class="subsection">
                <h3>Performance Metrics</h3>
                <table class="data-table">
                    <tr><th>Metric</th><th>Value</th><th>Units</th></tr>
                    <tr><td>Net Present Value (NPV)</td><td>${final_metrics.get('npv', 0):,.0f}</td><td>USD</td></tr>
                    <tr><td>Recovery Factor</td><td>{final_metrics.get('recovery_factor', 0):.2f}</td><td>%</td></tr>
                    <tr><td>CO2 Utilization</td><td>{final_metrics.get('co2_utilization', 0):.3f}</td><td>bbl/ton</td></tr>
                    <tr><td>CO2 Storage Efficiency</td><td>{final_metrics.get('co2_storage_efficiency', 0):.3f}</td><td>ton/bbl</td></tr>
                </table>
            </div>
            
            <div class="subsection">
                <h3>Optimization Method</h3>
                <p>Method used: {results.get('method', 'N/A')}</p>
                <p>Objective function value: {results.get('objective_function_value', 'N/A'):.6f}</p>
            </div>
            
            {charts_html}
        </div>
        <div class="page-break"></div>
        """

    def _generate_sensitivity_analysis(self, report_data: Dict[str, Any], config: Dict[str, Any]) -> str:
        """Generate sensitivity analysis HTML"""
        sensitivity_results = report_data.get('sensitivity_results', {})

        charts_html = ""
        if config.get('format_options', {}).get('include_charts', True) and 'charts' in report_data:
            charts = report_data['charts']
            if 'sensitivity_tornado' in charts:
                charts_html += f"""
                <div class="chart-container">
                    <h4>Sensitivity Analysis (Tornado Chart)</h4>
                    <img src="{charts['sensitivity_tornado']}" alt="Sensitivity Analysis Tornado Chart" class="chart-image">
                </div>
                """
        
        return f"""
        <div id="sensitivity-analysis" class="section">
            <h2>Sensitivity Analysis</h2>
            
            <div class="subsection">
                <h3>Key Sensitivity Parameters</h3>
                <table class="data-table">
                    <tr><th>Parameter</th><th>Sensitivity Index</th><th>Impact on NPV</th></tr>
                    {"".join([f'<tr><td>{param}</td><td>{data.get("sensitivity_index", 0):.3f}</td><td>{data.get("impact", "N/A")}</td></tr>' for param, data in sensitivity_results.items() if isinstance(data, dict)])}
                </table>
            </div>
            
            <div class="subsection">
                <h3>Analysis Summary</h3>
                <p>Sensitivity analysis conducted using {sensitivity_results.get('method', 'one-at-a-time')} method.</p>
                <p>Most sensitive parameters: {", ".join(list(sensitivity_results.keys())[:3]) if sensitivity_results else "N/A"}</p>
            </div>
            
            {charts_html}
        </div>
        <div class="page-break"></div>
        """

    def _generate_uncertainty_quantification(self, report_data: Dict[str, Any], config: Dict[str, Any]) -> str:
        """Generate uncertainty quantification HTML"""
        uq_results = report_data.get('uq_results', {})

        charts_html = ""
        if config.get('format_options', {}).get('include_charts', True) and 'charts' in report_data:
            charts = report_data['charts']
            if 'uq_distribution' in charts:
                charts_html += f"""
                <div class="chart-container">
                    <h4>Uncertainty Distribution</h4>
                    <img src="{charts['uq_distribution']}" alt="Uncertainty Distribution Chart" class="chart-image">
                </div>
                """
        
        return f"""
        <div id="uq-analysis" class="section">
            <h2>Uncertainty Quantification</h2>
            
            <div class="subsection">
                <h3>Uncertainty Analysis Results</h3>
                <table class="data-table">
                    <tr><th>Metric</th><th>P10</th><th>P50</th><th>P90</th></tr>
                    <tr><td>NPV (USD)</td><td>{uq_results.get('npv_p10', 0):,.0f}</td><td>{uq_results.get('npv_p50', 0):,.0f}</td><td>{uq_results.get('npv_p90', 0):,.0f}</td></tr>
                    <tr><td>Recovery Factor (%)</td><td>{uq_results.get('rf_p10', 0):.2f}</td><td>{uq_results.get('rf_p50', 0):.2f}</td><td>{uq_results.get('rf_p90', 0):.2f}</td></tr>
                </table>
            </div>
            
            <div class="subsection">
                <h3>Risk Assessment</h3>
                <p>Probability of economic success: {uq_results.get('success_probability', 0):.1%}</p>
                <p>Expected value: ${uq_results.get('expected_value', 0):,.0f}</p>
                <p>Standard deviation: ${uq_results.get('standard_deviation', 0):,.0f}</p>
            </div>
            
            {charts_html}
        </div>
        <div class="page-break"></div>
        """

    def _generate_economic_analysis(self, report_data: Dict[str, Any], config: Dict[str, Any]) -> str:
        """Generate economic analysis HTML"""
        economic_params = report_data.get('economic_parameters', {})
        optimization_results = report_data.get('optimization_results', {})
        final_metrics = optimization_results.get('final_metrics', {})

        charts_html = ""
        if config.get('format_options', {}).get('include_charts', True) and 'charts' in report_data:
            charts = report_data['charts']
            if 'economic_breakdown' in charts:
                charts_html += f"""
                <div class="chart-container">
                    <h4>Economic Breakdown</h4>
                    <img src="{charts['economic_breakdown']}" alt="Economic Breakdown Chart" class="chart-image">
                </div>
                """
        
        return f"""
        <div id="economic-assumptions" class="section">
            <h2>Economic Assumptions</h2>
            
            <div class="subsection">
                <h3>Economic Parameters</h3>
                <table class="data-table">
                    <tr><th>Parameter</th><th>Value</th><th>Units</th></tr>
                    <tr><td>Oil Price</td><td>${economic_params.get('oil_price_usd_per_bbl', 0):.2f}</td><td>USD/bbl</td></tr>
                    <tr><td>CO2 Purchase Cost</td><td>${economic_params.get('co2_purchase_cost_usd_per_tonne', 0):.2f}</td><td>USD/tonne</td></tr>
                    <tr><td>Discount Rate</td><td>{economic_params.get('discount_rate_fraction', 0):.1%}</td><td>-</td></tr>
                    <tr><td>CAPEX</td><td>${economic_params.get('capex_usd', 0):,.0f}</td><td>USD</td></tr>
                </table>
            </div>
            
            <div class="subsection">
                <h3>Economic Performance</h3>
                <table class="data-table">
                    <tr><th>Metric</th><th>Value</th><th>Units</th></tr>
                    <tr><td>Net Present Value (NPV)</td><td>${final_metrics.get('npv', 0):,.0f}</td><td>USD</td></tr>
                    <tr><td>Internal Rate of Return (IRR)</td><td>{final_metrics.get('irr', 0):.1%}</td><td>-</td></tr>
                    <tr><td>Payback Period</td><td>{final_metrics.get('payback_period', 0):.1f}</td><td>years</td></tr>
                </table>
            </div>
            
            <div class="subsection">
                <h3>Cost Breakdown</h3>
                <table class="data-table">
                    <tr><th>Cost Category</th><th>Amount</th><th>Percentage</th></tr>
                    <tr><td>CAPEX</td><td>${economic_params.get('capex_usd', 0):,.0f}</td><td>100%</td></tr>
                    <tr><td>OPEX</td><td>${final_metrics.get('total_opex', 0):,.0f}</td><td>{(final_metrics.get('total_opex', 0) / economic_params.get('capex_usd', 1) * 100) if economic_params.get('capex_usd', 0) > 0 else 0:.1f}%</td></tr>
                </table>
            </div>
            
            {charts_html}
        </div>
        <div class="page-break"></div>
        """

    def _generate_decline_curve_analysis(self, report_data: Dict[str, Any], config: Dict[str, Any]) -> str:
        dca_results = report_data.get('dca_results', {})

        charts_html = ""
        if config.get('format_options', {}).get('include_charts', True) and 'charts' in report_data:
            charts = report_data['charts']
            if 'dca_forecast' in charts:
                charts_html += f"""
                <div class="chart-container">
                    <h4>Production Forecast with Decline Curve Analysis</h4>
                    <img src="{charts['dca_forecast']}" alt="DCA Forecast Chart" class="chart-image">
                </div>
                """
            if 'dca_parameters' in charts:
                charts_html += f"""
                <div class="chart-container">
                    <h4>Decline Curve Parameters</h4>
                    <img src="{charts['dca_parameters']}" alt="DCA Parameters Chart" class="chart-image">
                </div>
                """

        model_type = dca_results.get('model_type', 'N/A')
        parameters = dca_results.get('parameters', {})
        r_squared = dca_results.get('r_squared', 0)
        economic_life = dca_results.get('economic_life', 0)
        ultimate_recovery = dca_results.get('ultimate_recovery', 0)
        peak_rate = dca_results.get('peak_rate', 0)
        
        return f"""
        <div id="decline-curve-analysis" class="section">
            <h2>Decline Curve Analysis</h2>
            
            <div class="subsection">
                <h3>Analysis Results</h3>
                <table class="data-table">
                    <tr><th>Parameter</th><th>Value</th><th>Units</th></tr>
                    <tr><td>Model Type</td><td>{model_type}</td><td>-</td></tr>
                    <tr><td>Initial Rate (qi)</td><td>{parameters.get('qi', 0):.2f}</td><td>{self.units_manager.get_unit('flow_rate')}</td></tr>
                    <tr><td>Decline Rate (di)</td><td>{parameters.get('di', 0):.4f}</td><td>1/year</td></tr>
                    <tr><td>Decline Exponent (b)</td><td>{parameters.get('b', 0):.3f}</td><td>-</td></tr>
                    <tr><td>R-squared</td><td>{r_squared:.4f}</td><td>-</td></tr>
                    <tr><td>Economic Life</td><td>{economic_life:.1f}</td><td>years</td></tr>
                    <tr><td>Ultimate Recovery</td><td>{ultimate_recovery:,.0f}</td><td>{self.units_manager.get_unit('volume')}</td></tr>
                    <tr><td>Peak Rate</td><td>{peak_rate:.2f}</td><td>{self.units_manager.get_unit('flow_rate')}</td></tr>
                </table>
            </div>
            
            <div class="subsection">
                <h3>Forecast Summary</h3>
                <p>Decline curve analysis indicates an economic life of {economic_life:.1f} years with an ultimate recovery of {ultimate_recovery:,.0f} {self.units_manager.get_unit('volume')}.</p>
                <p>The {model_type} decline model provides a good fit to the production data with an R-squared value of {r_squared:.4f}.</p>
            </div>
            
            {charts_html}
        </div>
        <div class="page-break"></div>
        """

    def _generate_decline_curve_analysis(self, report_data: Dict[str, Any], config: Dict[str, Any]) -> str:
        """Generate decline curve analysis HTML"""
        dca_results = report_data.get('dca_results', {})

        charts_html = ""
        if config.get('format_options', {}).get('include_charts', True) and 'charts' in report_data:
            charts = report_data['charts']
            if 'dca_forecast' in charts:
                charts_html += f"""
                <div class="chart-container">
                    <h4>Production Forecast with Decline Curve Analysis</h4>
                    <img src="{charts['dca_forecast']}" alt="DCA Forecast Chart" class="chart-image">
                </div>
                """
            if 'dca_parameters' in charts:
                charts_html += f"""
                <div class="chart-container">
                    <h4>Decline Curve Parameters</h4>
                    <img src="{charts['dca_parameters']}" alt="DCA Parameters Chart" class="chart-image">
                </div>
                """

        model_type = dca_results.get('model_type', 'N/A')
        parameters = dca_results.get('parameters', {})
        r_squared = dca_results.get('r_squared', 0)
        economic_life = dca_results.get('economic_life', 0)
        ultimate_recovery = dca_results.get('ultimate_recovery', 0)
        peak_rate = dca_results.get('peak_rate', 0)
        
        return f"""
        <div id="decline-curve-analysis" class="section">
            <h2>Decline Curve Analysis</h2>
            
            <div class="subsection">
                <h3>Analysis Results</h3>
                <table class="data-table">
                    <tr><th>Parameter</th><th>Value</th><th>Units</th></tr>
                    <tr><td>Model Type</td><td>{model_type}</td><td>-</td></tr>
                    <tr><td>Initial Rate (qi)</td><td>{parameters.get('qi', 0):.2f}</td><td>{self.units_manager.get_unit('flow_rate')}</td></tr>
                    <tr><td>Decline Rate (di)</td><td>{parameters.get('di', 0):.4f}</td><td>1/year</td></tr>
                    <tr><td>Decline Exponent (b)</td><td>{parameters.get('b', 0):.3f}</td><td>-</td></tr>
                    <tr><td>R-squared</td><td>{r_squared:.4f}</td><td>-</td></tr>
                    <tr><td>Economic Life</td><td>{economic_life:.1f}</td><td>years</td></tr>
                    <tr><td>Ultimate Recovery</td><td>{ultimate_recovery:,.0f}</td><td>{self.units_manager.get_unit('volume')}</td></tr>
                    <tr><td>Peak Rate</td><td>{peak_rate:.2f}</td><td>{self.units_manager.get_unit('flow_rate')}</td></tr>
                </table>
            </div>
            
            <div class="subsection">
                <h3>Forecast Summary</h3>
                <p>Decline curve analysis indicates an economic life of {economic_life:.1f} years with an ultimate recovery of {ultimate_recovery:,.0f} {self.units_manager.get_unit('volume')}.</p>
                <p>The {model_type} decline model provides a good fit to the production data with an R-squared value of {r_squared:.4f}.</p>
            </div>
            
            {charts_html}
        </div>
        <div class="page-break"></div>
        """

    def _generate_validation_report(self, report_data: Dict[str, Any]) -> str:
        """Generate model validation report section HTML"""
        validation_report_str = report_data.get('validation_report', 'Validation report not available.')
        return f"""
        <div id="validation-report" class="section">
            <h2>Model Validation</h2>
            <pre>{validation_report_str}</pre>
        </div>
        <div class="page-break"></div>
        """

    def _generate_appendices(self, report_data: Dict[str, Any]) -> str:
        """Generate appendices HTML"""
        return f"""
        <div id="appendices" class="section">
            <h2>Appendices</h2>
            
            <div class="subsection">
                <h3>Methodology</h3>
                <p>This report was generated using the CO2EOR Optimizer software, which employs advanced optimization algorithms and reservoir engineering principles to evaluate CO2 enhanced oil recovery strategies.</p>
            </div>
        </div>
        """

    def _get_css_styles(self, config: Dict[str, Any]) -> str:
        """Return CSS styles for the report with configurable formatting"""
        format_options = config.get('format_options', {})
        font_family = format_options.get('font_family', 'Times New Roman')
        font_size = format_options.get('font_size', 14)
        line_spacing = format_options.get('line_spacing', 1.0)
        
        return f"""
        <style>
            body {{
                font-family: "{font_family}", Times, serif;
                font-size: {font_size}px;
                line-height: {line_spacing};
                color: #000000;
                margin: 0;
                padding: 20px;
                background-color: #ffffff;
            }}
            
            .report-container {{
                max-width: 1000px;
                margin: 0 auto;
            }}
            
            .cover-page {{
                text-align: center;
                padding: 100px 0;
            }}
            
            .cover-page h1 {{
                font-size: {font_size + 4}px;
                color: #000000;
                margin-bottom: 20px;
            }}
            
            .cover-page h2 {{
                font-size: {font_size + 2}px;
                color: #666666;
                margin-bottom: 50px;
            }}
            
            .cover-details {{
                margin-top: 100px;
                text-align: left;
                max-width: 300px;
                margin-left: auto;
                margin-right: auto;
            }}
            
            .section {{
                margin-bottom: 40px;
            }}
            
            .section h2 {{
                color: #000000;
                border-bottom: 2px solid #000000;
                padding-bottom: 10px;
                margin-bottom: 20px;
                font-size: {font_size + 2}px;
            }}
            
            .subsection {{
                margin-bottom: 30px;
            }}
            
            .subsection h3 {{
                color: #000000;
                margin-bottom: 15px;
                font-size: {font_size + 1}px;
            }}
            
            .data-table {{
                width: 100%;
                border-collapse: collapse;
                margin-bottom: 20px;
                border: 2px solid #000000;
                font-size: {font_size}px;
            }}
            
            .data-table th, .data-table td {{
                border: 1px solid #000000;
                padding: 8px 12px;
                text-align: left;
            }}
            
            .data-table th {{
                background-color: #000000;
                color: #ffffff;
                font-weight: bold;
            }}
            
            .data-table tr:nth-child(even) {{
                background-color: #f2f2f2;
            }}
            
            .key-findings {{
                background-color: #f8f9fa;
                padding: 20px;
                border: 1px solid #000000;
                margin-bottom: 20px;
            }}
            
            .summary-metrics {{
                display: flex;
                justify-content: space-around;
                margin: 20px 0;
            }}
            
            .metric {{
                text-align: center;
                padding: 15px;
                background-color: white;
                border: 1px solid #000000;
                min-width: 120px;
            }}
            
            .metric-value {{
                display: block;
                font-size: {font_size + 10}px;
                font-weight: bold;
                color: #000000;
            }}
            
            .metric-label {{
                display: block;
                font-size: {font_size}px;
                color: #666666;
                margin-top: 5px;
            }}
            
            .recommendations {{
                margin-top: 20px;
            }}
            
            .recommendations ul {{
                padding-left: 20px;
            }}
            
            .recommendations li {{
                margin-bottom: 8px;
            }}
            
            .table-of-contents ul {{
                list-style-type: none;
                padding-left: 0;
            }}
            
            .table-of-contents li {{
                margin-bottom: 8px;
            }}
            
            .table-of-contents a {{
                color: #000000;
                text-decoration: none;
            }}
            
            .table-of-contents a:hover {{
                text-decoration: underline;
            }}
            
            .page-break {{
                page-break-after: always;
            }}
            
            pre {{
                background-color: #f8f9fa;
                padding: 15px;
                border: 1px solid #000000;
                overflow-x: auto;
                font-size: {font_size - 2}px;
            }}
            
            .chart-container {{
                margin: 20px 0;
                text-align: center;
            }}
            
            .chart-image {{
                max-width: 100%;
                height: auto;
                border: 1px solid #000000;
                padding: 10px;
                background-color: white;
            }}
        </style>
        """

    def _plotly_fig_to_base64(self, fig: go.Figure) -> str:
        """Convert a Plotly figure to base64 encoded image"""
        try:
            img_bytes = to_image(fig, format='png', width=800, height=600)
            return f"data:image/png;base64,{base64.b64encode(img_bytes).decode('utf-8')}"
        except Exception as e:
            self.error_occurred.emit(f"Failed to convert plot to image: {str(e)}")
            return ""

    def _matplotlib_fig_to_base64(self, fig: Figure) -> str:
        """Convert a Matplotlib figure to base64 encoded image"""
        try:
            canvas = FigureCanvasAgg(fig)
            buf = io.BytesIO()
            canvas.print_figure(buf, format='png', dpi=100)
            buf.seek(0)
            return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"
        except Exception as e:
            self.error_occurred.emit(f"Failed to convert matplotlib figure to image: {str(e)}")
            return ""