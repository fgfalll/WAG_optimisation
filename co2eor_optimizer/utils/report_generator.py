import os
import pdfkit
from PyQt6.QtCore import QObject, pyqtSignal
from .units_manager import UnitsManager

class ReportGenerator(QObject):
    progress_updated = pyqtSignal(int, str)
    report_generated = pyqtSignal(str)
    error_occurred = pyqtSignal(str)

    def __init__(self, units_manager: UnitsManager):
        super().__init__()
        self.units_manager = units_manager
        self.wkhtml_path = self.find_wkhtmltopdf()

    def find_wkhtmltopdf(self):
        """Find the path to wkhtmltopdf executable"""
        try:
            # First try the default binary location
            if os.path.exists("wkhtmltopdf/bin/wkhtmltopdf.exe"):
                return os.path.abspath("wkhtmltopdf/bin/wkhtmltopdf.exe")
            
            # Then check if it's in PATH
            import shutil
            path = shutil.which("wkhtmltopdf")
            if path:
                return path
                
            # Finally check common installation locations
            common_paths = [
                "C:/Program Files/wkhtmltopdf/bin/wkhtmltopdf.exe",
                "C:/Program Files (x86)/wkhtmltopdf/bin/wkhtmltopdf.exe"
            ]
            for path in common_paths:
                if os.path.exists(path):
                    return path
                    
            return None
        except:
            return None

    def generate_report(self, html_content, output_path, css_string=None):
        try:
            self.progress_updated.emit(0, "Starting report generation")
            
            # Configure pdfkit options
            options = {
                'enable-local-file-access': None,
                'quiet': '',
                'margin-top': '10mm',
                'margin-right': '10mm',
                'margin-bottom': '10mm',
                'margin-left': '10mm',
                'encoding': "UTF-8",
            }
            
            # Add CSS if provided
            if css_string:
                # Create a temporary CSS file
                css_path = os.path.join(os.path.dirname(output_path), "temp_style.css")
                with open(css_path, "w", encoding="utf-8") as css_file:
                    css_file.write(css_string)
                options['user-style-sheet'] = css_path
            
            # Configure pdfkit with found executable path
            config = pdfkit.configuration(wkhtmltopdf=self.wkhtml_path) if self.wkhtml_path else None
            
            self.progress_updated.emit(30, "Converting HTML to PDF")
            pdfkit.from_string(
                html_content, 
                output_path, 
                options=options, 
                configuration=config
            )
            
            # Clean up temporary CSS file
            if css_string and os.path.exists(css_path):
                os.remove(css_path)
                
            self.progress_updated.emit(100, "Report generated successfully")
            self.report_generated.emit(output_path)
            return True
        except Exception as e:
            self.progress_updated.emit(0, f"Error: {str(e)}")
            self.error_occurred.emit(f"PDF generation failed: {str(e)}")
            return False