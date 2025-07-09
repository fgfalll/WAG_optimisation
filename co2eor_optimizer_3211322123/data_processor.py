import logging
from pathlib import Path
from typing import Dict, List, Union, Callable, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

# Assuming the project structure from the architecture plan
from co2eor_optimizer.parsers.las_parser import parse_las
from co2eor_optimizer.parsers.eclipse_parser import EclipseParser, EclipseParserError
from co2eor_optimizer.core.data_models import WellData, ReservoirData
from co2eor_optimizer.utils.grdecl_writer import write_grdecl

logger = logging.getLogger(__name__)

class DataProcessor:
    """
    Processes a given list of input data files, parses them concurrently,
    and aggregates the results into structured data models.
    This version includes explicit validation of parser outputs and robust
    aggregation for complex reservoir models.
    """

    def __init__(self, filepaths: List[Union[str, Path]]):
        """
        Initializes the DataProcessor with a specific list of files.

        Args:
            filepaths: A list of full paths to the files to be processed.
        """
        self.filepaths = [Path(fp) for fp in filepaths]
        self._parser_map: Dict[str, Callable[..., Any]] = {
            '.las': parse_las,
            '.data': self._parse_eclipse_file,
            '.inc': self._parse_eclipse_file,
        }
        self.well_data_list: List[WellData] = []
        self.reservoir_data_list: List[ReservoirData] = []
        self.processed_files_count = 0
        self.failed_files: Dict[str, str] = {}  # Key: filename, Value: error message

    def _parse_eclipse_file(self, file_path: Path) -> Optional[ReservoirData]:
        """
        Wrapper method to parse an Eclipse file using the EclipseParser class.
        This isolates the instantiation of the parser from the dispatch logic.
        """
        parser = EclipseParser()
        try:
            return parser.parse(file_path)
        except EclipseParserError as e:
            logger.error(f"Eclipse parsing failed for {file_path.name}: {e}")
            # The exception will be caught and handled in _process_single_file
            raise

    def _process_single_file(self, file_path: Path) -> Optional[Union[WellData, ReservoirData]]:
        """
        Processes a single file by dispatching it to the correct parser.
        Returns the parsed data object or None if parsing is unsuccessful
        or the file type is unsupported.
        """
        ext = file_path.suffix.lower()
        parser_func = self._parser_map.get(ext)
        if not parser_func:
            # This case should ideally be filtered out before calling this method
            logger.warning(f"No parser available for file extension '{ext}' in {file_path.name}. Skipping.")
            return None

        logger.info(f"Dispatching parser for {ext.upper()} file: {file_path.name}")
        try:
            # The parser function itself is responsible for handling file IO and parsing logic.
            # It should return a data model instance or None on a "silent" failure (e.g., empty file).
            return parser_func(file_path)
        except Exception as e:
            # This catches critical errors during parsing (e.g., malformed file, IO errors).
            logger.error(f"Parser for file {file_path.name} raised an unhandled exception: {e}", exc_info=True)
            # Re-raise the exception to be caught and logged in the main processing loop.
            raise

    def _aggregate_reservoir_data(self) -> Optional[ReservoirData]:
        """
        Merges multiple ReservoirData objects into a single, comprehensive one.
        This is key for handling ECLIPSE projects with multiple INCLUDE files.
        Later-parsed files can update or add data to the base model.
        """
        if not self.reservoir_data_list:
            return None
        
        if len(self.reservoir_data_list) == 1:
            return self.reservoir_data_list[0]

        logger.info(f"Aggregating data from {len(self.reservoir_data_list)} reservoir source files.")
        # Start with the first parsed reservoir data as the base
        merged_data = self.reservoir_data_list[0]

        for subsequent_data in self.reservoir_data_list[1:]:
            # Merge grid properties
            if subsequent_data.grid:
                merged_data.grid.update(subsequent_data.grid)
            
            # Merge PVT tables
            if subsequent_data.pvt_tables:
                merged_data.pvt_tables.update(subsequent_data.pvt_tables)

            # Merge regions
            if subsequent_data.regions:
                if not merged_data.regions:
                    merged_data.regions = {}
                merged_data.regions.update(subsequent_data.regions)
            
            # Update runspec if the base doesn't have one
            if subsequent_data.runspec and not merged_data.runspec:
                merged_data.runspec = subsequent_data.runspec
            
            # **INTEGRATION POINT for EOS Model**
            # If the base model doesn't have an EOS definition yet, but a subsequent file does, adopt it.
            # This assumes the first file with compositional data defines the model for the whole case.
            if subsequent_data.eos_model and not merged_data.eos_model:
                logger.info(f"Adopting EOS model definition from a subsequent file.")
                merged_data.eos_model = subsequent_data.eos_model
        
        logger.info("Reservoir data aggregation complete.")
        return merged_data

    def process_files(
        self, progress_callback: Optional[Callable[[int, str], None]] = None
    ) -> Dict[str, Any]:
        """
        Processes all supported files from the initial list concurrently using a thread pool.

        Args:
            progress_callback: An optional function to call with progress updates.
                               It receives (progress_percent, message).

        Returns:
            A dictionary containing the processed data:
            - "well_data": A list of WellData objects.
            - "reservoir_data": A single, aggregated ReservoirData object or None.
            - "failed_files": A dictionary of filenames and their corresponding error messages.
        """
        files_to_process = [
            p for p in self.filepaths if p.suffix.lower() in self._parser_map
        ]
        
        if not files_to_process:
            logger.warning("No supported files (.las, .data, .inc) found in the provided list.")
            if progress_callback:
                progress_callback(100, "No supported files to process.")
            return {"well_data": [], "reservoir_data": None, "failed_files": {}}

        total_files = len(files_to_process)
        
        with ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as executor:
            future_to_file = {
                executor.submit(self._process_single_file, file): file
                for file in files_to_process
            }

            for future in as_completed(future_to_file):
                file = future_to_file[future]
                self.processed_files_count += 1

                try:
                    result = future.result()
                    
                    # --- EXPLICIT VALIDATION AND SORTING OF PARSER OUTPUT ---
                    if result is None:
                        # Case 1: The parser intentionally returned None (e.g., empty file, non-relevant content)
                        logger.warning(f"Parser for {file.name} returned no data.")
                        # This isn't necessarily a failure, but we can log it if needed.
                        # self.failed_files[file.name] = "File is empty or lacks required data sections."
                        continue

                    if isinstance(result, WellData):
                        self.well_data_list.append(result)
                    elif isinstance(result, ReservoirData):
                        # Case 2: A ReservoirData object was returned, but it might be empty.
                        if not result.grid and not result.pvt_tables and not result.runspec:
                            logger.warning(f"Reservoir file {file.name} parsed but contained no actionable data (grid, pvt, etc.).")
                            self.failed_files[file.name] = "File parsed but contained no valid grid, PVT, or runspec data."
                        else:
                            self.reservoir_data_list.append(result)
                    else:
                        logger.warning(f"Processing {file.name} returned an unexpected data type: {type(result).__name__}")
                    
                except Exception as e:
                    # Case 3: The parser raised a critical exception.
                    logger.error(f"A critical exception occurred while processing file {file.name}: {e}")
                    self.failed_files[file.name] = f"Processing error: {e}"

                finally:
                    if progress_callback:
                        progress = int(100 * self.processed_files_count / total_files)
                        message = f"Processed: {file.name}"
                        progress_callback(progress, message)

        final_reservoir_data = self._aggregate_reservoir_data()

        return {
            "well_data": self.well_data_list,
            "reservoir_data": final_reservoir_data,
            "failed_files": self.failed_files,
        }

    def get_summary(self) -> Dict[str, Any]:
        """Returns a summary of the processed data after running process_files."""
        final_reservoir = self._aggregate_reservoir_data()
        
        reservoir_summary: Dict[str, Any] = {
            "total_source_files": len(self.reservoir_data_list),
            "has_grid_data": False,
            "grid_keywords": [],
            "dims": 'N/A'
        }
        
        if final_reservoir:
            reservoir_summary["has_grid_data"] = bool(final_reservoir.grid)
            if final_reservoir.grid:
                 reservoir_summary['grid_keywords'] = list(final_reservoir.grid.keys())
                 reservoir_summary['dims'] = final_reservoir.grid.get('DIMENS') or final_reservoir.grid.get('SPECGRID_DIMS', 'N/A')

        return {
            'total_files_processed': self.processed_files_count,
            'wells': {'total_valid': len(self.well_data_list), 'names': [w.name for w in self.well_data_list]},
            'reservoir': reservoir_summary,
            'failures': {'count': len(self.failed_files), 'files': list(self.failed_files.keys())}
        }