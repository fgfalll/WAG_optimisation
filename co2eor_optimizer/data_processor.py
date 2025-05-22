import os
from pathlib import Path
from typing import Dict, List, Union
import logging
from concurrent.futures import ThreadPoolExecutor
from .parsers.las_parser import parse_las
from .parsers.eclipse_parser import parse_eclipse
from .core import WellData, ReservoirData

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataProcessor:
    """Processes input data files from specified directory"""
    
    def __init__(self, input_dir: str = "Import files"):
        self.input_dir = Path(input_dir)
        self.well_data: List[WellData] = []
        self.reservoir_data: List[ReservoirData] = []
        
    def process_files(self) -> Dict[str, Union[List[WellData], List[ReservoirData]]]:
        """Process all files in input directory"""
        if not self.input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {self.input_dir}")
            
        files = list(self.input_dir.glob("*"))
        if not files:
            logger.warning(f"No files found in directory: {self.input_dir}")
            return {
                'well_data': [],
                'reservoir_data': []
            }
            
        with ThreadPoolExecutor() as executor:
            futures = []
            for file in files:
                futures.append(executor.submit(self._process_single_file, file))
                
            for future in futures:
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Failed to process file: {str(e)}")
                    
        return {
            'well_data': self.well_data,
            'reservoir_data': self.reservoir_data
        }
        
    def _process_single_file(self, file_path: Path) -> None:
        """Process a single file based on its extension"""
        ext = file_path.suffix.lower()
        
        try:
            if ext == '.las':
                logger.info(f"Processing LAS file: {file_path}")
                well_data = parse_las(str(file_path))
                self.well_data.append(well_data)
            elif ext == '.data':
                logger.info(f"Processing Eclipse file: {file_path}")
                try:
                    reservoir_data = parse_eclipse(str(file_path))
                    if reservoir_data and reservoir_data.grid:
                        self.reservoir_data.append(reservoir_data)
                    else:
                        logger.warning(f"Empty or invalid data in {file_path}")
                except Exception as e:
                    logger.error(f"Failed to parse {file_path}: {e}", exc_info=True)
            else:
                logger.warning(f"Skipping unsupported file type: {file_path}")
        except Exception as e:
            logger.error(f"Error processing {file_path}: {str(e)}")
            raise
            
    def get_summary(self) -> Dict[str, Dict[str, int]]:
        """Get summary of processed data"""
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
    """Main entry point for command line usage"""
    processor = DataProcessor()
    try:
        results = processor.process_files()
        summary = processor.get_summary()
        
        print("\nProcessing complete!")
        print(f"Processed {summary['wells']['total']} well files")
        print(f"Processed {summary['reservoirs']['total']} reservoir files")
        
        return results
    except Exception as e:
        logger.error(f"Fatal error during processing: {str(e)}")
        raise

if __name__ == "__main__":
    main()