"""
SR3 Parser Wrapper for CMG GEM Output

This module wraps the sr3_reader.py module to parse CMG SR3 binary output files.

NOTE: This module performs NO CALCULATIONS.
Only reads and parses binary output from CMG.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import sys

logger = logging.getLogger(__name__)

try:
    from validation.sr3_reader import read_SR3, get_sector_timeseries, search_acronym
except ImportError:
    try:
        from .sr3_reader import read_SR3, get_sector_timeseries, search_acronym
    except ImportError:
        _validation_dir = str(Path(__file__).parent)
        if _validation_dir not in sys.path:
            sys.path.insert(0, _validation_dir)
        try:
            from sr3_reader import read_SR3, get_sector_timeseries, search_acronym
        except ImportError as e:
            logger.error(f"Failed to import sr3_reader: {e}")
            logger.error("Make sure h5py is installed: pip install h5py")
            raise


class SR3Parser:
    """
    Parse CMG SR3 binary output files.

    NOTE: This class performs NO CALCULATIONS.
    Only reads and parses binary output from CMG.

    Example:
        >>> parser = SR3Parser()
        >>> results = parser.parse_file(Path("case_output/gmflu002.sr3"))
        >>> print(f"Oil production: {results['cumulative_oil']:.0f} STB")
    """

    def __init__(self):
        """Initialize SR3 parser."""
        pass

    def parse_file(self, sr3_file: Any) -> Dict[str, Any]:
        """
        Parse CMG SR3 binary output file.

        NOTE: This function performs NO CALCULATIONS.
        Only reads and parses binary output from CMG.

        Args:
            sr3_file: Path to .sr3 file (string or Path object)

        Returns:
            Dictionary with parsed results including time-series profiles
        """
        # Ensure sr3_file is a Path object
        if isinstance(sr3_file, str):
            sr3_file = Path(sr3_file)
            
        results = {
            # Cumulative values (final)
            'cumulative_oil': 0.0,
            'cumulative_gas': 0.0,
            'cumulative_water': 0.0,
            'cumulative_co2_injected': 0.0,
            'cumulative_co2_produced': 0.0,
            # Scalar values
            'final_pressure': 0.0,
            'recovery_factor': 0.0,
            'ooip_stb': 0.0,
            # Time-series profiles (arrays)
            'time_vector': np.array([]),
            'pressure_profile': np.array([]),
            'oil_production_rate': np.array([]),
            'oil_cumulative_profile': np.array([]),
            'gas_production_rate': np.array([]),
            'gas_cumulative_profile': np.array([]),
            'water_production_rate': np.array([]),
            'recovery_profile': np.array([]),
            # Oil in place profile
            'oil_in_place_profile': np.array([]),
            # Raw sector timeseries dataframe
            'sector_timeseries': None,
            # Status
            'warnings': [],
            'errors': [],
        }

        if not sr3_file.exists():
            results['errors'].append(f"SR3 file not found: {sr3_file}")
            return results

        try:
            # Read SR3 file
            logger.info(f"Reading SR3 file: {sr3_file}")
            sr3 = read_SR3(str(sr3_file), timeout=120)

            # Get sector timeseries (FIELD level)
            ts = get_sector_timeseries(sr3)
            results['sector_timeseries'] = ts

            if len(ts) == 0:
                results['warnings'].append("No sector timeseries found in SR3 file")
                return results

            final_row = ts.iloc[-1]

            # --- Time vector ---
            if 'Days' in ts.columns:
                results['time_vector'] = ts['Days'].values.astype(float)

            # --- Pressure profile ---
            if 'PDTVSEC' in ts.columns:
                results['pressure_profile'] = ts['PDTVSEC'].values.astype(float)
                results['final_pressure'] = float(ts['PDTVSEC'].iloc[-1])

            # --- Oil production ---
            # OILSECPRCM = cumulative oil produced (ft3) -> Convert to STB
            # OILSECPRRT = instantaneous oil production rate (ft3/day) -> Convert to STB/day
            # OILSECSU   = oil remaining in sector (ft3) - NOT cumulative production!
            if 'OILSECPRCM' in ts.columns:
                results['oil_cumulative_profile'] = ts['OILSECPRCM'].values.astype(float) / 5.61458
                results['cumulative_oil'] = float(ts['OILSECPRCM'].iloc[-1]) / 5.61458

            if 'OILSECPRRT' in ts.columns:
                results['oil_production_rate'] = ts['OILSECPRRT'].values.astype(float) / 5.61458
            # Oil in place (sector)
            if 'OILSECSU' in ts.columns:
                results['oil_in_place_profile'] = ts['OILSECSU'].values.astype(float)

            # --- Gas production ---
            # GASSECPRCM = cumulative gas produced (SCF)
            # GASSECPRRT = instantaneous gas production rate (SCF/day)
            if 'GASSECPRCM' in ts.columns:
                results['gas_cumulative_profile'] = ts['GASSECPRCM'].values.astype(float)
                results['cumulative_gas'] = float(ts['GASSECPRCM'].iloc[-1])

            if 'GASSECPRRT' in ts.columns:
                results['gas_production_rate'] = ts['GASSECPRRT'].values.astype(float)

            # --- Water production ---
            # WATSECPRCM = cumulative water produced (STB)
            # WATSECPRRT = instantaneous water production rate (STB/day)
            if 'WATSECPRCM' in ts.columns:
                results['cumulative_water'] = float(ts['WATSECPRCM'].iloc[-1])

            if 'WATSECPRRT' in ts.columns:
                results['water_production_rate'] = ts['WATSECPRRT'].values.astype(float)

            # --- CO2/Solvent injection ---
            # INLSECPRCM = cumulative solvent injection (SCF)
            # GASSECINCM = cumulative gas injection (SCF)
            if 'INLSECPRCM' in ts.columns:
                inj = float(ts['INLSECPRCM'].iloc[-1])
                if inj > 0:
                    results['cumulative_co2_injected'] = inj
            if results['cumulative_co2_injected'] == 0 and 'GASSECINCM' in ts.columns:
                results['cumulative_co2_injected'] = float(ts['GASSECINCM'].iloc[-1])

            # --- Recovery factor ---
            # OILSECRECOO = original oil recovery factor (%)
            # OILSECRECO  = oil recovery factor (%)
            if 'OILSECRECOO' in ts.columns:
                results['recovery_factor'] = float(ts['OILSECRECOO'].iloc[-1]) / 100.0
                results['recovery_profile'] = ts['OILSECRECOO'].values.astype(float) / 100.0
            elif 'OILSECRECO' in ts.columns:
                results['recovery_factor'] = float(ts['OILSECRECO'].iloc[-1]) / 100.0
                results['recovery_profile'] = ts['OILSECRECO'].values.astype(float) / 100.0

            # --- OOIP ---
            # Calculate from cumulative oil and recovery factor
            if results['cumulative_oil'] > 0 and results['recovery_factor'] > 0:
                results['ooip_stb'] = results['cumulative_oil'] / results['recovery_factor']

            # Check for missing expected data
            if results['cumulative_oil'] == 0.0:
                results['warnings'].append("Cumulative oil production not found in SR3 file")
            if results['ooip_stb'] == 0.0:
                results['warnings'].append("OOIP not found/calculated from SR3 file")

            logger.info(f"  Parsed {len(ts)} timesteps, {len(ts.columns)} variables")
            logger.info(f"  Cumulative oil: {results['cumulative_oil']:,.0f} STB")
            logger.info(f"  Recovery factor: {results['recovery_factor']:.2%}")
            logger.info(f"  Final pressure: {results['final_pressure']:.1f} psia")

        except Exception as e:
            logger.error(f"Error parsing SR3 file: {e}")
            import traceback
            traceback.print_exc()
            results['errors'].append(f"SR3 parsing error: {e}")

        return results

    def parse_directory(self, output_dir: Path) -> Dict[str, Any]:
        """
        Parse SR3 file in a directory.

        NOTE: This function performs NO CALCULATIONS.
        Only reads and parses binary output from CMG.

        Args:
            output_dir: Directory containing CMG output files

        Returns:
            Dictionary with parsed results
        """
        if not output_dir.exists():
            return {
                'cumulative_oil': 0.0,
                'ooip_stb': 0.0,
                'errors': [f"Output directory not found: {output_dir}"]
            }

        # Look for .sr3 files
        sr3_files = list(output_dir.glob("*.sr3"))

        if not sr3_files:
            return {
                'cumulative_oil': 0.0,
                'ooip_stb': 0.0,
                'errors': [f"No SR3 files found in {output_dir}"]
            }

        # Use the first SR3 file found
        return self.parse_file(sr3_files[0])

    def list_available_variables(self, sr3_file: Path) -> None:
        """
        Print available variables in SR3 file.

        Args:
            sr3_file: Path to .sr3 file
        """
        try:
            sr3 = read_SR3(str(sr3_file), timeout=30)
            ts = get_sector_timeseries(sr3)

            print(f"\nAvailable variables in {sr3_file.name}:")
            print("-" * 60)
            for col in ts.columns:
                print(f"  {col}")
            print("-" * 60)
            print(f"Total time steps: {len(ts)}")

        except Exception as e:
            print(f"Error listing variables: {e}")


__all__ = ["SR3Parser"]
