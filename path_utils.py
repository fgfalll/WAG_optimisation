"""
Path utilities for CO2 EOR Optimizer.

Provides functions to handle resource paths correctly in both
development mode and frozen (PyInstaller) mode.
"""

from pathlib import Path
import sys


def get_app_root() -> Path:
    """
    Get the application root directory.

    Returns the executable directory when frozen, or the package root
    when running in development mode.

    Returns:
        Path: Application root directory
    """
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent
    return Path(__file__).parent


def get_resource_path(*path_parts) -> Path:
    """
    Get the path to a bundled resource file.

    When frozen, resources are in _MEIPASS or alongside the executable.
    When in development, resources are in the package root.

    Args:
        *path_parts: Path components to join (e.g., "config", "base_config.json")

    Returns:
        Path: Full path to the resource file
    """
    if getattr(sys, "frozen", False):
        base = Path(sys._MEIPASS) if hasattr(sys, "_MEIPASS") else Path(sys.executable).parent
    else:
        base = Path(__file__).parent

    return base.joinpath(*path_parts)


def get_config_dir() -> Path:
    """
    Get the configuration directory path.

    Returns:
        Path: Configuration directory
    """
    return get_resource_path("config")


def get_help_dir() -> Path:
    """
    Get the help documentation directory path.

    Returns:
        Path: Help directory
    """
    return get_resource_path("help")


def get_translations_dir() -> Path:
    """
    Get the translations directory path.

    Returns:
        Path: Translations directory
    """
    return get_resource_path("translations")


def get_logs_dir() -> Path:
    """
    Get the logs directory path.

    In frozen mode, logs are stored alongside the executable.
    In development mode, logs are stored in the package root.

    Returns:
        Path: Logs directory
    """
    return get_app_root() / "logs"


def get_ui_assets_dir() -> Path:
    """
    Get the UI assets directory path (icons, images, etc.).

    Returns:
        Path: UI assets directory
    """
    return get_resource_path("ui", "assets")


def get_mathjax_dir() -> Path:
    """
    Get the MathJax directory path.

    Returns:
        Path: MathJax directory
    """
    return get_resource_path("ui", "dialogs", "mathjax")
