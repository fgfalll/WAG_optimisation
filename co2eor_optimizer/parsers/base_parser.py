import abc
import logging
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Union, List, Tuple
from time import perf_counter

from co2eor_optimizer.core.data_models import ReservoirData, WellData
from co2eor_optimizer.parsers.validation import validate_reservoir_data, validate_well_data

class ParserError(Exception):
    """Custom exception for parser-related errors"""
    pass

class BaseParser(metaclass=abc.ABCMeta):
    """
    Abstract base class for all data parsers in the CO₂ EOR optimization suite.
    Defines the common interface and implements shared functionality.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.reset_state()
        
    def reset_state(self):
        """Reset parser state between files"""
        self.file_path = None
        self.raw_data = None
        self.parse_time = 0.0
        self.validation_errors = []
        
    @abc.abstractmethod
    def parse(self, file_path: Union[str, Path]) -> bool:
        """
        Parse the specified file. Must be implemented by subclasses.
        Returns True on success, False on failure.
        """
        pass
    
    @abc.abstractmethod
    def to_reservoir_data(self) -> Optional[ReservoirData]:
        """Convert parsed data to ReservoirData object"""
        pass
    
    @abc.abstractmethod
    def to_well_data(self) -> Optional[WellData]:
        """Convert parsed data to WellData object"""
        pass
    
    def validate(self, strict: bool = True) -> bool:
        """
        Validate parsed data against schema and business rules.
        Returns True if valid, False otherwise.
        """
        reservoir_data = self.to_reservoir_data()
        well_data = self.to_well_data()
        
        self.validation_errors = []
        
        if reservoir_data:
            res_errors = validate_reservoir_data(reservoir_data)
            if res_errors:
                self.validation_errors.extend(res_errors)
        
        if well_data:
            well_errors = validate_well_data(well_data)
            if well_errors:
                self.validation_errors.extend(well_errors)
        
        if strict and self.validation_errors:
            self.logger.error(f"Validation failed with {len(self.validation_errors)} errors")
            return False
            
        return len(self.validation_errors) == 0
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Return performance metrics for the last parse operation"""
        return {
            "parse_time_sec": self.parse_time,
            "file_size_mb": Path(self.file_path).stat().st_size / (1024 * 1024) if self.file_path else 0
        }
    
    def get_validation_report(self) -> List[str]:
        """Get detailed validation error report"""
        return self.validation_errors
    
    def _log_parse_start(self, file_path: Union[str, Path]):
        """Standardized parse operation logging"""
        self.file_path = Path(file_path)
        self.logger.info(f"Starting parse of {self.file_path.name}")
        self.raw_data = None
        self.parse_time = perf_counter()
        
    def _log_parse_end(self, success: bool):
        """Standardized parse completion logging"""
        self.parse_time = perf_counter() - self.parse_time
        status = "SUCCESS" if success else "FAILED"
        self.logger.info(f"Parse {status} for {self.file_path.name} in {self.parse_time:.2f}s")