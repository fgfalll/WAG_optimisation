import os
from pathlib import Path
from typing import Dict, List, Union, Callable, Any
import logging
from concurrent.futures import ThreadPoolExecutor
from .parsers.las_parser import parse_las
from .parsers.eclipse_parser import parse_eclipse
from core.data_models import WellData, ReservoirData
from .utils.grdecl_writer import write_grdecl

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataProcessor:
    """Processes input data files from a specified directory."""

    def __init__(self, input_dir: str = "import"):
        """
        Initializes the DataProcessor.

        Args:
            input_dir: The directory containing input files.
        """
        self.input_dir = Path(input_dir)
        self.well_data: List[WellData] = []
        self.reservoir_data: List[ReservoirData] = []
        # A dictionary to map file extensions to their respective parser functions.
        # This makes the processor more extensible.
        self._parser_map: Dict[str, Callable[[str], Any]] = {
            '.las': self._parse_and_store_las,
            '.data': self._parse_and_store_eclipse,
        }

    def _parse_and_store_las(self, file_path: str) -> None:
        """Helper to parse a LAS file and store its data."""
        well_data = parse_las(file_path)
        self.well_data.append(well_data)

    def _parse_and_store_eclipse(self, file_path: str) -> None:
        """Helper to parse an Eclipse file and store its data."""
        reservoir_data = parse_eclipse(file_path)
        if reservoir_data and reservoir_data.grid:
            # Store the original filename stem for more descriptive output naming
            setattr(reservoir_data, 'source_filename', Path(file_path).stem)
            self.reservoir_data.append(reservoir_data)
        else:
            logger.warning(f"No valid grid data found in {file_path}")

    def process_files(self) -> Dict[str, Union[List[WellData], List[ReservoirData]]]:
        """
        Processes all supported files in the input directory concurrently.
        """
        if not self.input_dir.is_dir():
            raise FileNotFoundError(f"Input directory not found: {self.input_dir}")

        files = list(self.input_dir.glob("*"))
        if not files:
            logger.warning(f"No files found in directory: {self.input_dir}")
            return {'well_data': [], 'reservoir_data': []}

        with ThreadPoolExecutor() as executor:
            # Submit files to be processed by _process_single_file
            future_to_file = {executor.submit(self._process_single_file, file): file for file in files}

            for future in future_to_file:
                file = future_to_file[future]
                try:
                    future.result()  # result() will re-raise exceptions from the worker thread
                except Exception as e:
                    # Enhanced logging includes the exception type for more context.
                    logger.error(f"Failed to process file {file}: {type(e).__name__} - {e}")

        # Write GRDECL files for all processed reservoir data
        self._write_reservoir_data()

        return {
            'well_data': self.well_data,
            'reservoir_data': self.reservoir_data
        }

    def _write_reservoir_data(self):
        """Writes all collected reservoir data to GRDECL files."""
        for reservoir in self.reservoir_data:
            if reservoir.grid:
                # Improved naming for the output file.
                source_name = getattr(reservoir, 'source_filename', 'reservoir_grid')
                output_path = self.input_dir / f"{source_name}.grdecl"
                try:
                    write_grdecl(reservoir, output_path)
                    logger.info(f"Successfully wrote GRDECL file: {output_path}")
                except Exception as e:
                    logger.error(f"Failed to write GRDECL file {output_path}: {type(e).__name__} - {e}")

    def _process_single_file(self, file_path: Path) -> None:
        """
        Processes a single file by dispatching it to the correct parser
        based on its extension.
        """
        ext = file_path.suffix.lower()
        parser_func = self._parser_map.get(ext)

        if parser_func:
            logger.info(f"Processing {ext.upper()} file: {file_path}")
            parser_func(str(file_path))
        else:
            logger.warning(f"Skipping unsupported file type: {file_path}")

    def get_summary(self) -> Dict[str, Dict[str, int]]:
        """
        Returns a summary of the processed data.
        """
        return {
            'wells': {
                'total': len(self.well_data),
                'with_data': sum(1 for wd in self.well_data if wd.properties)
            },
            'reservoirs': {
                'total': len(self.reservoir_data),
                'with_data': sum(1 for rd in self.reservoir_data if rd.grid)
            }
        }

def main():
    """Main entry point for command-line usage."""
    try:
        processor = DataProcessor()
        processor.process_files()
        summary = processor.get_summary()

        print("\n--- Processing Summary ---")
        print(f"Processed {summary['wells']['with_data']}/{summary['wells']['total']} well files with data.")
        print(f"Processed {summary['reservoirs']['with_data']}/{summary['reservoirs']['total']} reservoir files with grid data.")
        print("--------------------------")

    except FileNotFoundError as e:
        logger.error(f"Directory not found: {e}")
    except Exception as e:
        logger.critical(f"A fatal error occurred during processing: {e}", exc_info=True)

if __name__ == "__main__":
    main()