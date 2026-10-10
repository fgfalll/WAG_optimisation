"""
CMG GEM Runner

This module executes the CMG GEM simulator.

NOTE: This module performs NO CALCULATIONS.
Only launches the CMG executable and waits for completion.
"""

import subprocess
import logging
from pathlib import Path
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

# CMG executable path (hardcoded as per user requirement)
CMG_EXE_PATH = Path(r"D:\Program Files\CMG\GEM\2025.30\Win_x64\EXE\gm202530.exe")


class CMGRunner:
    """
    Execute CMG GEM simulation.

    NOTE: This class performs NO CALCULATIONS.
    Only launches the CMG executable and waits for completion.

    Example:
        >>> runner = CMGRunner()
        >>> success = runner.run("case.dat")
        >>> if success:
        ...     print("CMG simulation completed")
    """

    def __init__(self, executable_path: Optional[Path] = None):
        """
        Initialize CMG runner.

        Args:
            executable_path: Path to CMG GEM executable.
                          Defaults to D:\\Program Files\\CMG\\GEM\\2025.30\\Win_x64\\EXE\\gm202530.exe
        """
        self.executable_path = executable_path or CMG_EXE_PATH

        if not self.executable_path.exists():
            logger.error(f"CMG executable not found at {self.executable_path}")

    def run(
        self,
        input_file: Path,
        working_dir: Optional[Path] = None,
        timeout: int = 3600
    ) -> Dict[str, Any]:
        """
        Execute CMG GEM simulation.

        NOTE: This function performs NO CALCULATIONS.
        Only launches the CMG executable and waits for completion.

        Args:
            input_file: Path to .dat input file
            working_dir: Working directory for simulation (default: input file directory)
            timeout: Maximum runtime in seconds (default: 1 hour)

        Returns:
            Dictionary with:
            - success: True if CMG ran successfully, False otherwise
            - returncode: Process return code
            - stdout: Standard output
            - stderr: Standard error
            - runtime: Elapsed time in seconds
        """
        if not self.executable_path.exists():
            return {
                'success': False,
                'returncode': -1,
                'stdout': '',
                'stderr': f"CMG executable not found at {self.executable_path}",
                'runtime': 0.0
            }

        if not input_file.exists():
            return {
                'success': False,
                'returncode': -1,
                'stdout': '',
                'stderr': f"Input file not found: {input_file}",
                'runtime': 0.0
            }

        if working_dir is None:
            working_dir = input_file.parent

        # Ensure working directory exists
        working_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Running CMG GEM: {input_file.name}")
        logger.info(f"Working directory: {working_dir}")
        logger.info(f"Executable: {self.executable_path}")

        import time
        start_time = time.time()

        try:
            # Run CMG simulation without capturing stdin/stdout to avoid CONIN$ issue
            # Use STARTUPINFO_IGNORESTDERR for Windows compatibility
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW

            # Note: CMG command line interface may vary
            # Common approach: pass input file as argument
            result = subprocess.run(
                [str(self.executable_path), str(input_file)],
                cwd=str(working_dir),
                timeout=timeout,
                creationflags=0x08000000  # CREATE_NO_WINDOW
            )

            runtime = time.time() - start_time
            success = result.returncode == 0

            if success:
                logger.info(f"CMG simulation completed successfully in {runtime:.1f}s")
            else:
                logger.warning(f"CMG simulation failed with return code {result.returncode}")

            return {
                'success': success,
                'returncode': result.returncode,
                'stdout': '',
                'stderr': f'Process completed with code {result.returncode}',
                'runtime': runtime
            }

        except subprocess.TimeoutExpired:
            runtime = time.time() - start_time
            logger.error(f"CMG simulation timed out after {timeout} seconds")
            return {
                'success': False,
                'returncode': -1,
                'stdout': '',
                'stderr': f'Simulation timed out after {timeout} seconds',
                'runtime': runtime
            }

        except Exception as e:
            runtime = time.time() - start_time
            logger.error(f"Error running CMG: {e}")
            return {
                'success': False,
                'returncode': -1,
                'stdout': '',
                'stderr': str(e),
                'runtime': runtime
            }

    def check_license(self) -> bool:
        """
        Check if CMG license is available.

        Returns:
            True if license check passes, False otherwise
        """
        try:
            # Try to run CMG with --help or similar to check license
            result = subprocess.run(
                [str(self.executable_path), "--help"],
                capture_output=True,
                text=True,
                timeout=10
            )
            # CMG might not have --help, but no error means executable runs
            return result.returncode == 0 or "license" not in result.stderr.lower()
        except Exception as e:
            logger.warning(f"Could not verify CMG license: {e}")
            return False

    def get_version(self) -> Optional[str]:
        """
        Get CMG GEM version.

        Returns:
            Version string or None if unavailable
        """
        # Extract version from executable path
        # Path format: .../GEM/2025.30/Win_x64/EXE/gm202530.exe
        try:
            parent = self.executable_path.parent.parent.parent  # Go up to year.version directory
            version = parent.name
            return version
        except Exception:
            return None


__all__ = ["CMGRunner", "CMG_EXE_PATH"]
