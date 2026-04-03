from PyQt6.QtCore import QObject, pyqtSignal, QThread
from typing import Any, Dict

class DCAWorker(QThread):
    """Worker thread for running Decline Curve Analysis."""
    def __init__(self, analyzer, well_data, model_type, forecast_years):
        super().__init__()
        self.analyzer = analyzer
        self.well_data = well_data
        self.model_type = model_type
        self.forecast_years = forecast_years
        self.signals = self.WorkerSignals()

    class WorkerSignals(QObject):
        finished = pyqtSignal()
        error = pyqtSignal(str)
        result = pyqtSignal(object)

    def run(self):
        try:
            time = self.well_data.properties.get("time")
            rate = self.well_data.properties.get("rate")

            if time is None or rate is None:
                raise ValueError("Production data (time and rate) not found in well data properties.")

            result = self.analyzer.analyze_production(
                time=time,
                production_rate=rate,
                model_type=self.model_type,
                forecast_years=self.forecast_years,
            )
            self.signals.result.emit(result)
        except Exception as e:
            self.signals.error.emit(str(e))
        finally:
            self.signals.finished.emit()
