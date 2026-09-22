"""Custom exceptions for CO2 EOR Optimizer - Fail-Fast error handling."""


class SurrogateEngineError(Exception):
    """Raised when surrogate engine evaluation fails critically."""

    pass


class OptimizationError(Exception):
    """Raised when optimization evaluation fails."""

    pass


class RecoveryModelError(Exception):
    """Raised when recovery factor calculation fails."""

    pass


class ProfileGenerationError(Exception):
    """Raised when profile generation fails."""

    pass


class SimulationEngineError(Exception):
    """Raised when simulation engine evaluation fails."""

    pass


class DeclineCurveAnalysisError(Exception):
    """Raised when decline curve analysis fails."""

    pass


class SensitivityAnalysisError(Exception):
    """Raised when sensitivity analysis fails."""

    pass