"""
CMG Output Parser

This module parses CMG GEM output files.

NOTE: This module performs NO CALCULATIONS.
Only reads and parses text output from CMG.
"""

from typing import Dict, Any, List, Optional
from pathlib import Path
import numpy as np
import logging
import re

logger = logging.getLogger(__name__)


class CMGOutputParser:
    """
    Parse CMG GEM output files.

    NOTE: This class performs NO CALCULATIONS.
    Only reads and parses text output from CMG.

    Example:
        >>> parser = CMGOutputParser()
        >>> results = parser.parse_directory(Path("case_output"))
        >>> print(f"Oil production: {results['cumulative_oil']:.0f} STB")
    """

    def __init__(self):
        """Initialize CMG output parser."""
        pass

    def parse_directory(self, output_dir: Path) -> Dict[str, Any]:
        """
        Parse all CMG output files in a directory.

        NOTE: This function performs NO CALCULATIONS.
        Only reads and parses text output from CMG.

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
            'oil_production_rate': np.array([]),
            'gas_production_rate': np.array([]),
            'water_production_rate': np.array([]),
            'time_vector': np.array([]),
            'pressure_profile': np.array([]),
            'final_pressure': 0.0,
            'recovery_factor': 0.0,
            'warnings': [],
            'errors': [],
        }

        if not output_dir.exists():
            results['errors'].append(f"Output directory not found: {output_dir}")
            return results

        # Try multiple file naming conventions
        # 1. Directory name as prefix: output_dir/output_dir.mrt
        # 2. Common prefixes: gmflu002, case, data
        possible_prefixes = [
            output_dir.name,
            "gmflu002",
            "case",
            "data",
        ]

        mrt_file = None
        out_file = None

        for prefix in possible_prefixes:
            mrt_candidate = output_dir / f"{prefix}.mrt"
            out_candidate = output_dir / f"{prefix}.out"
            if mrt_candidate.exists():
                mrt_file = mrt_candidate
            if out_candidate.exists():
                out_file = out_candidate

        # Also try listing all files and finding matches
        if mrt_file is None or out_file is None:
            for file in output_dir.glob("*"):
                if file.suffix == ".mrt" and mrt_file is None:
                    mrt_file = file
                elif file.suffix == ".out" and out_file is None:
                    out_file = file

        if mrt_file:
            logger.info(f"Parsing MRT file: {mrt_file}")
            try:
                results.update(self._parse_mrt_file(mrt_file))
            except Exception as e:
                logger.warning(f"Failed to parse MRT file: {e}")
                results['warnings'].append(f"MRT parsing failed: {e}")

        if out_file:
            logger.info(f"Parsing OUT file: {out_file}")
            try:
                results.update(self._parse_out_file(out_file))
            except Exception as e:
                logger.warning(f"Failed to parse OUT file: {e}")
                results['warnings'].append(f"OUT parsing failed: {e}")

        if not mrt_file and not out_file:
            results['errors'].append(f"No CMG output files found in {output_dir}")

        return results

    def _parse_mrt_file(self, mrt_file: Path) -> Dict[str, Any]:
        """
        Parse CMG .mrt results file.

        NOTE: This function performs NO CALCULATIONS.
        Only reads and parses text output from CMG.

        Args:
            mrt_file: Path to .mrt file

        Returns:
            Dictionary with parsed results
        """
        results = {
            'time_vector': np.array([]),
            'oil_production_rate': np.array([]),
            'water_production_rate': np.array([]),
            'gas_production_rate': np.array([]),
            'cumulative_oil': 0.0,
            'cumulative_water': 0.0,
            'cumulative_gas': 0.0,
        }

        try:
            with open(mrt_file, 'r') as f:
                content = f.read()

            # Parse time series data using regex
            # CMG format varies, this is a simplified parser

            # Look for common result keywords
            patterns = {
                'FOPT': r'FOPT\s*[\d.]+\s*[\d.]+\s*([\d.]+)',  # Field Oil Production Total
                'FWPT': r'FWPT\s*[\d.]+\s*[\d.]+\s*([\d.]+)',  # Field Water Production Total
                'FGPT': r'FGPT\s*[\d.]+\s*[\d.]+\s*([\d.]+)',  # Field Gas Production Total
            }

            for key, pattern in patterns.items():
                match = re.search(pattern, content)
                if match:
                    value = float(match.group(1))
                    if key == 'FOPT':
                        results['cumulative_oil'] = value
                    elif key == 'FWPT':
                        results['cumulative_water'] = value
                    elif key == 'FGPT':
                        results['cumulative_gas'] = value

        except Exception as e:
            logger.error(f"Error parsing MRT file: {e}")

        return results

    def _parse_out_file(self, out_file: Path) -> Dict[str, Any]:
        """
        Parse CMG .out output file.

        NOTE: This function performs NO CALCULATIONS.
        Only reads and parses text output from CMG.

        Args:
            out_file: Path to .out file

        Returns:
            Dictionary with parsed results
        """
        results = {
            'final_pressure': 0.0,
            'ooip_mstb': 0.0,
            'cumulative_oil_mstb': 0.0,
            'cumulative_gas_mmscf': 0.0,
            'cumulative_water_mstb': 0.0,
            'cumulative_co2_injected_mmscf': 0.0,
            'recovery_factor_pct': 0.0,
            'oil_production_rate_mstb_day': 0.0,
            'gas_production_rate_mmscf_day': 0.0,
            'warnings': [],
            'errors': [],
        }

        try:
            with open(out_file, 'r', errors='ignore') as f:
                content = f.read()

            # Parse final summary section (last occurrence)
            # Look for "Originally in Place" section for OOIP
            ooip_pattern = r'Stock Tank Oil\s+M STB\s+([\d.]+)'
            matches = re.findall(ooip_pattern, content)
            if matches:
                results['ooip_mstb'] = float(matches[0])  # First occurrence is OOIP

            # Look for "Cumulative Production" section (last occurrence)
            # Format: "Oil              M STB      XXXXX"
            cumulative_section = content.rsplit('Cumulative Production', 1)[-1].split('Injection Rate')[0] if 'Cumulative Production' in content else ''
            if cumulative_section:
                oil_match = re.search(r'Oil\s+M STB\s+([\d.]+)', cumulative_section)
                if oil_match:
                    results['cumulative_oil_mstb'] = float(oil_match.group(1))

                gas_match = re.search(r'Gas\s+MM SCF\s+([\d.]+)', cumulative_section)
                if gas_match:
                    results['cumulative_gas_mmscf'] = float(gas_match.group(1))

                water_match = re.search(r'Water\s+M STB\s+([\d. E]+)', cumulative_section)
                if water_match:
                    # Handle scientific notation like "1.77149E-7"
                    val = water_match.group(1).replace(' ', '').replace('E', 'e')
                    try:
                        results['cumulative_water_mstb'] = float(val)
                    except:
                        pass

            # Look for "Cumulative Injection" section for CO2 injected
            injection_section = content.rsplit('Cumulative Injection', 1)[-1].split('Cumulative Production')[0] if 'Cumulative Injection' in content else ''
            if injection_section:
                co2_match = re.search(r'Solvent\s+MM SCF\s+([\d.]+)', injection_section)
                if co2_match:
                    results['cumulative_co2_injected_mmscf'] = float(co2_match.group(1))

            # Parse recovery factor (RECO)
            reco_pattern = r'RECO\)\s*=\s*([\d.]+)'
            reco_matches = re.findall(reco_pattern, content)
            if reco_matches:
                results['recovery_factor_pct'] = float(reco_matches[-1])  # Last occurrence

            # Parse average pressure
            pressure_pattern = r'Total PV Ave\.\s+psia\s+([\d.]+)'
            pressure_matches = re.findall(pressure_pattern, content)
            if pressure_matches:
                results['final_pressure'] = float(pressure_matches[-1])  # Last occurrence

            # Parse production rates
            rate_section = content.rsplit('Production Rate', 1)[-1].split('Injection Rate')[0] if 'Production Rate' in content else ''
            if rate_section:
                oil_rate_match = re.search(r'Oil\s+M STB/day\s+([\d.]+)', rate_section)
                if oil_rate_match:
                    results['oil_production_rate_mstb_day'] = float(oil_rate_match.group(1))

                gas_rate_match = re.search(r'Gas\s+MM SCF/day\s+([\d.]+)', rate_section)
                if gas_rate_match:
                    results['gas_production_rate_mmscf_day'] = float(gas_rate_match.group(1))

            # Convert to standard units (STB, SCF)
            results['cumulative_oil'] = results['cumulative_oil_mstb'] * 1000  # M STB to STB
            results['ooip_stb'] = results['ooip_mstb'] * 1000  # M STB to STB
            results['recovery_factor'] = results['recovery_factor_pct'] / 100.0  # % to fraction

            # Check for errors
            error_patterns = [
                r'ERROR',
                r'FATAL',
                r'CONVERGENCE FAILED',
            ]

            for pattern in error_patterns:
                if re.search(pattern, content, re.IGNORECASE):
                    results['errors'].append(f"Found error pattern: {pattern}")

            # Check for warnings
            warning_patterns = [
                r'WARNING',
                r'TIMESTEP CUT',
            ]

            for pattern in warning_patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                if matches:
                    results['warnings'].append(f"Found {len(matches)} occurrences of {pattern}")

        except Exception as e:
            logger.error(f"Error parsing OUT file: {e}")
            results['errors'].append(f"OUT parsing error: {e}")

        return results

    def parse_summary_file(self, summary_file: Path) -> Dict[str, Any]:
        """
        Parse CMG summary (.mrt) file for detailed time series.

        NOTE: This function performs NO CALCULATIONS.
        Only reads and parses text output from CMG.

        Args:
            summary_file: Path to summary file

        Returns:
            Dictionary with time series data
        """
        results = {
            'time_days': np.array([]),
            'oil_rate': np.array([]),
            'water_rate': np.array([]),
            'gas_rate': np.array([]),
            'pressure': np.array([]),
            'water_cut': np.array([]),
            'gor': np.array([]),
        }

        if not summary_file.exists():
            logger.error(f"Summary file not found: {summary_file}")
            return results

        try:
            with open(summary_file, 'r') as f:
                lines = f.readlines()

            # Parse data (simplified - CMG format is complex)
            # This is a placeholder for a full parser
            data_started = False
            time_data = []
            oil_data = []

            for line in lines:
                line = line.strip()
                if not line or line.startswith('*'):
                    if 'TIME' in line.upper():
                        data_started = True
                    continue

                if data_started:
                    # Try to parse numeric data
                    try:
                        values = re.findall(r'[\d.]+', line)
                        if len(values) >= 2:
                            time_data.append(float(values[0]))
                            oil_data.append(float(values[1]))
                    except ValueError:
                        pass

            if time_data:
                results['time_days'] = np.array(time_data)
                results['oil_rate'] = np.array(oil_data)

        except Exception as e:
            logger.error(f"Error parsing summary file: {e}")

        return results


__all__ = ["CMGOutputParser"]
