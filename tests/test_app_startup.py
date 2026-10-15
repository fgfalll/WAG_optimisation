"""
Smoke tests for application startup and UI/Analysis subsystem imports.
Verifies that all widgets, analyzers, and timed imports required by main.py
can be imported without ModuleNotFoundError or circular import deadlocks.
"""

import pytest
from PyQt6.QtWidgets import QApplication


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_main_window_timed_import(qapp):
    """Verify that main.timed_import_main_window imports MainWindow and all tab widgets."""
    from main import timed_import_main_window
    MainWindow = timed_import_main_window()
    assert MainWindow is not None


def test_sensitivity_analyzer_import():
    """Verify that SensitivityAnalyzer and all analytical dependencies import cleanly."""
    from analysis.sensitivity_analyzer import SensitivityAnalyzer
    assert SensitivityAnalyzer is not None


def test_production_profiler_import():
    """Verify that ProductionProfiler imports cleanly."""
    from analysis.profiler_refactored import ProductionProfiler
    assert ProductionProfiler is not None
