"""
Combined CMG Output Parser

This module combines data from both text (.out) and binary (.sr3) CMG output files.

NOTE: This module performs NO CALCULATIONS.
Only reads and parses output from CMG.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np

logger = logging.getLogger(__name__)

# Import existing parsers
from .cmg_output_parser import CMGOutputParser
try:
    from .sr3_parser import SR3Parser
    SR3_AVAILABLE = True
except ImportError:
    SR3_AVAILABLE = False
    logger.warning("SR3 parser not available. Install h5py: pip install h5py")


class CombinedCMGParser:
    """
    Parse CMG output from both text (.out) and binary (.sr3) files.

    NOTE: This class performs NO CALCULATIONS.
    Only reads and parses output from CMG.

    Uses .out file for cumulative/summary data (more reliable).
    Uses .sr3 file for time series data (pressure profiles, production rates).
    """

    def __init__(self):
        """Initialize combined CMG parser."""
        self.out_parser = CMGOutputParser()
        if SR3_AVAILABLE:
            self.sr3_parser = SR3Parser()
        else:
            self.sr3_parser = None

    def parse_directory(self, output_dir: Path) -> Dict[str, Any]:
        """
        Parse all CMG output files in a directory.

        NOTE: This function performs NO CALCULATIONS.
        Only reads and parses output from CMG.

        Args:
            output_dir: Directory containing CMG output files

        Returns:
            Dictionary with parsed results
        """
        results = {
            'cumulative_oil': 0.0,
            'cumulative_gas': 0.0,
            'cumulative_water': 0.0,
            'cumulative_co2_injected': 0.0,
            'cumulative_co2_produced': 0.0,
            'oil_production_rate': np.array([]),
            'gas_production_rate': np.array([]),
            'water_production_rate': np.array([]),
            'time_vector': np.array([]),
            'pressure_profile': np.array([]),
            'final_pressure': 0.0,
            'recovery_factor': 0.0,
            'ooip_stb': 0.0,
            'warnings': [],
            'errors': [],
        }

        if not output_dir.exists():
            results['errors'].append(f"Output directory not found: {output_dir}")
            return results

        # Parse .out file for summary data (more reliable)
        out_file = self._find_output_file(output_dir, '.out')
        if out_file:
            logger.info(f"Parsing OUT file: {out_file}")
            try:
                out_results = self.out_parser._parse_out_file(out_file)

                # Extract summary data from OUT file
                results['ooip_stb'] = out_results.get('ooip_mstb', 0) * 1000.0
                results['cumulative_oil'] = out_results.get('cumulative_oil_mstb', 0) * 1000.0
                results['cumulative_gas'] = out_results.get('cumulative_gas_mmscf', 0) * 1.0  # Already MMSCF
                results['cumulative_water'] = out_results.get('cumulative_water_mstb', 0) * 1000.0
                results['cumulative_co2_injected'] = out_results.get('cumulative_co2_injected_mmscf', 0) * 1.0  # Already MMSCF
                results['recovery_factor'] = out_results.get('recovery_factor_pct', 0) / 100.0
                results['final_pressure'] = out_results.get('final_pressure', 0)

                # Production rates from OUT file (final values)
                if out_results.get('oil_production_rate_mstb_day'):
                    results['oil_production_rate'] = np.array([out_results['oil_production_rate_mstb_day'] * 1000.0])
                if out_results.get('gas_production_rate_mmscf_day'):
                    results['gas_production_rate'] = np.array([out_results['gas_production_rate_mmscf_day']])

                # Add warnings/errors from OUT parser
                results['warnings'].extend(out_results.get('warnings', []))
                results['errors'].extend(out_results.get('errors', []))

                logger.info(f"  OOIP: {results['ooip_stb']:,.0f} STB")
                logger.info(f"  Cumulative Oil: {results['cumulative_oil']:,.0f} STB")
                logger.info(f"  Recovery Factor: {results['recovery_factor']:.2%}")

            except Exception as e:
                logger.error(f"Error parsing OUT file: {e}")
                results['errors'].append(f"OUT parsing error: {e}")

        # Parse .sr3 file for time series data (if available)
        if SR3_AVAILABLE and self.sr3_parser:
            sr3_file = self._find_output_file(output_dir, '.sr3')
            if sr3_file:
                logger.info(f"Parsing SR3 file for time series: {sr3_file}")
                try:
                    sr3_results = self.sr3_parser.parse_file(sr3_file)

                    # Extract time series data from SR3
                    if len(sr3_results.get('time_vector', [])) > 0:
                        results['time_vector'] = sr3_results['time_vector']
                        logger.info(f"  Time steps: {len(results['time_vector'])}")

                    # Extract pressure profile from SR3 (more detailed)
                    if len(sr3_results.get('pressure_profile', [])) > 0:
                        results['pressure_profile'] = sr3_results['pressure_profile']
                        logger.info(f"  Pressure profile: {len(results['pressure_profile'])} points")

                    # Use SR3 data for production profiles if available
                    if len(sr3_results.get('oil_production_profile', [])) > 0:
                        results['oil_production_profile'] = sr3_results['oil_production_profile']

                    # Add warnings/errors from SR3 parser
                    results['warnings'].extend(sr3_results.get('warnings', []))
                    results['errors'].extend(sr3_results.get('errors', []))

                except Exception as e:
                    logger.error(f"Error parsing SR3 file: {e}")
                    results['warnings'].append(f"SR3 parsing error: {e}")

        # Check for required data
        if results['cumulative_oil'] == 0.0:
            results['errors'].append("Cumulative oil production not found in output files")

        return results

    def _find_output_file(self, output_dir: Path, suffix: str) -> Optional[Path]:
        """Find output file with given suffix in directory."""
        # Try common prefixes
        possible_prefixes = [
            output_dir.name,
            "gmflu002",
            "case",
            "data",
        ]

        for prefix in possible_prefixes:
            candidate = output_dir / f"{prefix}{suffix}"
            if candidate.exists():
                return candidate

        # Use glob to find any file with the suffix
        matches = list(output_dir.glob(f"*{suffix}"))
        if matches:
            return matches[0]

        return None


__all__ = ["CombinedCMGParser"]
