import os
import re
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional, Any, List, Set

import numpy as np

from core.data_models import ReservoirData

# --- Setup Logging ---
# Setup a module-level logger for clear and consistent output.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# --- Constants for Eclipse Keywords ---
# Grouping keywords makes the parser's logic clearer and easier to maintain.
SECTION_KEYWORDS = ['RUNSPEC', 'GRID', 'PROPS', 'REGIONS', 'SOLUTION', 'SCHEDULE', 'SUMMARY']
DIMENS_KEYWORDS = ['DIMENS', 'SPECGRID']
IMPLICIT_GRID_KEYWORDS = ['DIMENS', 'SPECGRID', 'COORD', 'ZCORN']
PVT_KEYWORDS = {'PVTO', 'PVDO', 'PVTG', 'PVDG'} # Using a set for fast lookups
SAT_TABLE_KEYWORDS = {'SWFN', 'SGFN', 'SOF3', 'SWOF'}
REGION_KEYWORDS = {'EQLNUM', 'FIPNUM', 'SATNUM'}
ARRAY_MODIFICATION_KEYWORDS = ['COPY', 'ADD', 'MULTIPLY', 'EQUALS']
TERMINATOR = '/'

# --- Precompiled Regex Patterns for Performance ---
COMMENT_PATTERN = re.compile(r"--.*")
INCLUDE_PATTERN = re.compile(
    r"^\s*INCLUDE\s*['\"]([^'\"]+)['\"]",
    re.IGNORECASE | re.MULTILINE
)
KEYWORD_PATTERN = re.compile(r"^\s*([A-Z_][A-Z0-9_]*)\s*", re.IGNORECASE)


# --- Custom Exceptions for Clearer Error Reporting ---
class EclipseParserError(Exception):
    """Base exception for all parser-related errors."""
    pass

class IncludeProcessingError(EclipseParserError):
    """Raised for errors during INCLUDE file processing, like circular dependencies."""
    pass

class SectionParsingError(EclipseParserError):
    """Raised for errors during the parsing of a specific section."""
    pass

class DimensionError(EclipseParserError):
    """Raised for issues related to grid dimensions (e.g., not defined, incorrect values)."""
    pass


class EclipseParser:
    """
    A comprehensive parser for ECLIPSE 100 data files.

    This parser reads ECLIPSE data files, processes included sub-files,
    interprets keywords and data blocks, and structures the output. It is
    designed to be stateful during a single parse operation (to remember
    grid dimensions) but can be reset for reuse.
    """
    def __init__(self):
        """Initializes the parser."""
        self.reset()
        self.logger = logging.getLogger(__name__)

    def reset(self):
        """Resets the parser's state, allowing it to be reused."""
        self.dims: Optional[Tuple[int, int, int]] = None
        self.logger.info("EclipseParser has been reset.")

    @staticmethod
    def _expand_eclipse_values(data_tokens: List[str]) -> List[str]:
        """
        Expands the 'N*value' syntax into a list of repeated values.
        
        This version is more robust and handles cases like 'N*' at the end of a line.
        """
        expanded_values = []
        for token in data_tokens:
            if '*' not in token:
                expanded_values.append(token)
                continue

            try:
                count_str, value_str = token.split('*', 1)
                if count_str.isdigit():
                    count = int(count_str)
                    # If value is empty (e.g., '3* '), it implies repeating the next token.
                    # This case is handled by the main parsing loop logic. Here we assume '3*value'.
                    if value_str:
                        expanded_values.extend([value_str] * count)
                    else: # Handle dangling 'N*'
                        expanded_values.append(token) # Keep it for later processing
                else:
                    expanded_values.append(token)
            except ValueError:
                expanded_values.append(token)
        
        # Second pass to handle 'N* value' syntax
        final_values = []
        i = 0
        while i < len(expanded_values):
            token = expanded_values[i]
            if token.endswith('*') and token[:-1].isdigit():
                count = int(token[:-1])
                if i + 1 < len(expanded_values):
                    value_to_repeat = expanded_values[i+1]
                    final_values.extend([value_to_repeat] * count)
                    i += 2 # Consume both 'N*' and 'value'
                else:
                    # Dangling 'N*' at the very end of data
                    i += 1
            else:
                final_values.append(token)
                i += 1
        return final_values

    def _parse_pvt_or_sat_table(self, content: str, keyword: str) -> np.ndarray:
        """
        Parses generic multi-column tables like PVT or Saturation tables.
        This unified function simplifies parsing of all table-based keywords.
        """
        try:
            lines = [line.strip() for line in content.splitlines() if line.strip() and not line.strip() == TERMINATOR]
            
            table_data = []
            for line in lines:
                # Remove comments from the line before splitting
                cleaned_line = COMMENT_PATTERN.sub("", line).strip()
                if not cleaned_line:
                    continue

                raw_tokens = cleaned_line.split()
                expanded_tokens = self._expand_eclipse_values(raw_tokens)
                
                values = []
                for token in expanded_tokens:
                    try:
                        # Standardize scientific notation and convert to float
                        values.append(float(token.lower().replace('d', 'e')))
                    except (ValueError, TypeError):
                        self.logger.warning(f"Non-numeric value '{token}' in {keyword} table line: '{line}'. Skipping token.")
                        continue
                
                if values:
                    table_data.append(values)
            
            if not table_data:
                self.logger.warning(f"No valid data found for keyword {keyword} after parsing.")
                return np.array([])

            # Using dtype=object allows for jagged arrays if rows have different lengths
            return np.array(table_data, dtype=object)
        except Exception as e:
            self.logger.exception(f"Failed to parse table for keyword {keyword}")
            raise SectionParsingError(f"Table parsing failed for {keyword}: {e}") from e

    def _apply_array_modifications(self, data: Dict[str, Any], keyword: str, block_content: str):
        """
        Applies EQUALS, COPY, or MULTIPLY operations to the data dictionary.
        This is a refactoring of the old _preprocess_special_blocks logic.
        """
        if not self.dims:
            self.logger.warning(f"Found {keyword} block but grid dimensions (DIMENS) are not yet set. Skipping.")
            return

        nx, ny, nz = self.dims
        total_cells = nx * ny * nz

        lines = block_content.strip().splitlines()
        for line in lines:
            parts = COMMENT_PATTERN.sub("", line).strip().split()
            if not parts or parts[-1] == TERMINATOR:
                parts = [p for p in parts if p != TERMINATOR]
            if not parts:
                continue

            try:
                if keyword == 'EQUALS':
                    if len(parts) < 8: continue
                    target_kw, val_str = parts[0].upper(), parts[1]
                    value = float(val_str)
                    i1, i2, j1, j2, k1, k2 = map(int, parts[2:8])

                    if target_kw not in data:
                        # Initialize with NaN, which is a better default for missing data
                        data[target_kw] = np.full(total_cells, np.nan, dtype=np.float64)
                    
                    # Reshape to 3D grid using Fortran ordering for ECLIPSE compatibility
                    grid_view = data[target_kw].reshape((nz, ny, nx), order='F')
                    # Apply value to the specified slice (adjusting for 1-based indexing)
                    grid_view[k1-1:k2, j1-1:j2, i1-1:i2] = value
                    # The underlying array 'data[target_kw]' is modified in place
                
                elif keyword in ('COPY', 'MULTIPLY', 'ADD'):
                    if len(parts) < 2: continue
                    source_kw, dest_kw = parts[0].upper(), parts[1].upper()

                    if source_kw not in data:
                        self.logger.warning(f"Source keyword '{source_kw}' for {keyword} not found. Skipping line: '{line}'.")
                        continue
                    
                    source_array = data[source_kw]
                    if dest_kw not in data:
                        # For COPY/ADD, create a copy. For MULTIPLY, create an array of ones.
                        if keyword == 'MULTIPLY':
                           data[dest_kw] = np.ones_like(source_array)
                        else:
                           data[dest_kw] = np.copy(source_array)

                    if keyword == 'COPY':
                        data[dest_kw] = np.copy(source_array)
                    
                    else: # MULTIPLY or ADD
                        # Default is to apply to the whole grid
                        value = float(parts[2]) if len(parts) > 2 else 1.0
                        box = (0, nx, 0, ny, 0, nz)
                        if len(parts) >= 9: # Regional operation
                            box = map(int, parts[3:9])
                        i1, i2, j1, j2, k1, k2 = box
                        
                        grid_view = data[dest_kw].reshape((nz, ny, nx), order='F')
                        source_view = source_array.reshape((nz, ny, nx), order='F')
                        region_slice = (slice(k1-1, k2), slice(j1-1, j2), slice(i1-1, i2))

                        if keyword == 'MULTIPLY':
                           grid_view[region_slice] *= source_view[region_slice] * value
                        elif keyword == 'ADD':
                           grid_view[region_slice] += source_view[region_slice] * value # ADD has a factor

            except (ValueError, IndexError) as e:
                self.logger.warning(f"Could not parse line in {keyword} block: '{line}'. Error: {e}")

    def _parse_section(self, section_content: str, section_name: str) -> Dict[str, Any]:
        """
        Parses all keyword-data blocks within a single section's content.
        This is a more robust method that scans for keywords linearly.
        """
        parsed_data = {}
        content_ptr = 0

        while content_ptr < len(section_content):
            # Find the next keyword
            match = KEYWORD_PATTERN.search(section_content, content_ptr)
            if not match:
                break # No more keywords in the section

            keyword = match.group(1).upper()
            block_start = match.end()

            # Find the end of the block (start of next keyword or end of content)
            next_match = KEYWORD_PATTERN.search(section_content, block_start)
            block_end = next_match.start() if next_match else len(section_content)
            
            block_content = section_content[block_start:block_end]
            content_ptr = block_end # Move pointer for next iteration

            # Handle special array modification blocks first
            if keyword in ARRAY_MODIFICATION_KEYWORDS:
                self._apply_array_modifications(parsed_data, keyword, block_content)
                continue

            # Tokenize the data within the block
            lines = COMMENT_PATTERN.sub("", block_content).strip().splitlines()
            data_tokens = []
            for line in lines:
                # A line with only a slash is a definite terminator
                if line.strip() == TERMINATOR:
                    break
                
                parts = line.strip().split()
                # Check for terminator at the end of a token list
                if parts and parts[-1] == TERMINATOR:
                    parts.pop()
                    data_tokens.extend(parts)
                    break 
                data_tokens.extend(parts)
            
            if not data_tokens:
                parsed_data[keyword] = np.nan # Keyword present but no data
                continue
            
            # --- Handle different keyword types ---
            
            # Standardize PVT aliases on the fly
            if keyword == 'PVDO': keyword = 'PVTO'
            if keyword == 'PVDG': keyword = 'PVTG'

            if keyword in PVT_KEYWORDS or keyword in SAT_TABLE_KEYWORDS:
                parsed_data[keyword] = self._parse_pvt_or_sat_table(block_content, keyword)
                continue

            # Expand N*value syntax and convert to numeric types
            expanded_tokens = self._expand_eclipse_values(data_tokens)
            numeric_values = []
            for token in expanded_tokens:
                try:
                    numeric_values.append(float(token.lower().replace('d', 'e')))
                except (ValueError, TypeError):
                    # Keep non-numeric values as strings (e.g., for TITLE)
                    numeric_values.append(token.strip("'\""))

            if keyword in DIMENS_KEYWORDS:
                if len(numeric_values) < 3:
                    raise DimensionError(f"{keyword} requires at least 3 values, got {numeric_values}")
                nx, ny, nz = map(int, numeric_values[:3])
                self.dims = (nx, ny, nz)
                self.logger.info(f"Parsed grid dimensions: {self.dims} from {keyword}")
                parsed_data[keyword] = list(self.dims)
                continue
            
            # For grid properties, ensure array is correctly sized
            if section_name in ('GRID', 'PROPS', 'SOLUTION') and self.dims:
                is_grid_property = any(isinstance(v, (int, float)) for v in numeric_values)
                if is_grid_property:
                    expected_size = self.dims[0] * self.dims[1] * self.dims[2]
                    # Handle single-value expansion (e.g., PORO 0.2 /)
                    if len(numeric_values) == 1 and expected_size > 1:
                        numeric_values = [numeric_values[0]] * expected_size
                    # Pad or truncate if data size is incorrect
                    if len(numeric_values) < expected_size:
                        self.logger.warning(f"Keyword '{keyword}': expected {expected_size} values, got {len(numeric_values)}. Padding with NaN.")
                        numeric_values.extend([np.nan] * (expected_size - len(numeric_values)))
                    elif len(numeric_values) > expected_size:
                        self.logger.warning(f"Keyword '{keyword}': expected {expected_size} values, got {len(numeric_values)}. Truncating.")
                        numeric_values = numeric_values[:expected_size]

            try:
                parsed_data[keyword] = np.array(numeric_values, dtype=np.float64)
            except ValueError:
                parsed_data[keyword] = np.array(numeric_values, dtype=object)

        return parsed_data

    def _resolve_include_path(self, include_path: str, base_path: str) -> Path:
        """Resolves an INCLUDE file path, prioritizing relative paths to the base file."""
        base_dir = Path(base_path).parent
        
        # Path relative to the file containing the INCLUDE statement
        resolved_from_base = base_dir / include_path
        if resolved_from_base.exists():
            return resolved_from_base.resolve()
            
        # Fallback to path relative to current working directory
        resolved_from_cwd = Path.cwd() / include_path
        if resolved_from_cwd.exists():
            self.logger.debug(f"Resolved '{include_path}' relative to CWD, not file location '{base_path}'.")
            return resolved_from_cwd.resolve()
            
        # Return the original path if not found, to be handled by the caller
        return resolved_from_base.resolve()

    def _process_includes(self, content: str, base_file_path: str, visited: Optional[Set[Path]] = None) -> str:
        """Recursively processes INCLUDE statements, preventing circular dependencies."""
        if visited is None:
            visited = set()

        # The absolute path of the current file is added to the visited set
        current_abs_path = Path(base_file_path).resolve()
        visited.add(current_abs_path)

        def replace_include(match):
            path_str = match.group(1)
            resolved_path = self._resolve_include_path(path_str, base_file_path)

            if resolved_path in visited:
                self.logger.warning(f"Circular INCLUDE detected and skipped: {resolved_path}")
                return ""
            
            if not resolved_path.exists():
                self.logger.warning(f"INCLUDE file not found and skipped: {resolved_path}")
                return ""
            
            try:
                with resolved_path.open('r', errors='ignore') as f:
                    included_content = f.read()
                # Recursively process includes in the newly added content
                return self._process_includes(included_content, str(resolved_path), visited)
            except IOError as e:
                raise IncludeProcessingError(f"Error reading included file {resolved_path}: {e}") from e

        return INCLUDE_PATTERN.sub(replace_include, content)

    def _split_into_sections(self, content: str) -> Dict[str, str]:
        """Parses content into sections based on major keywords."""
        sections = {}
        section_splits = re.split(f"({'|'.join(SECTION_KEYWORDS)})", content, flags=re.IGNORECASE)
        
        # The first item is content before any section keyword
        implicit_content = section_splits[0]
        if any(kw in implicit_content.upper() for kw in IMPLICIT_GRID_KEYWORDS):
             sections['GRID'] = implicit_content
             self.logger.info("Implicit GRID section detected at the beginning of the file.")
        elif implicit_content.strip():
             sections['RUNSPEC'] = implicit_content
             self.logger.info("Treating content before first section keyword as RUNSPEC.")

        # Process the rest of the splits (keyword, content, keyword, content, ...)
        for i in range(1, len(section_splits), 2):
            section_keyword = section_splits[i].upper()
            section_content = section_splits[i+1]
            sections[section_keyword] = sections.get(section_keyword, "") + section_content

        return sections

def parse_eclipse(file_path: str) -> "ReservoirData":
    """
    Main function to parse an ECLIPSE 100 data file into a structured object.

    Args:
        file_path (str): The path to the main ECLIPSE data file.

    Returns:
        ReservoirData: A structured object containing the parsed data.
    
    Raises:
        FileNotFoundError: If the initial file_path does not exist.
        EclipseParserError: For any critical error during the parsing process.
    """
    parser = EclipseParser()
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"ECLIPSE data file not found: {path}")

    logger.info(f"Starting ECLIPSE file parsing: {path}")
    try:
        with path.open('r', errors='ignore') as f:
            initial_content = f.read()

        # 1. Process all INCLUDEs to get a single monolithic string
        full_content = parser._process_includes(initial_content, str(path))
        if not full_content.strip():
            raise EclipseParserError("ECLIPSE file is empty or contains only INCLUDEs to empty files.")
        
        # 2. Split the content into major sections
        sections = parser._split_into_sections(full_content)
        
        # 3. Parse each section individually
        runspec_data = parser._parse_section(sections.get('RUNSPEC', ''), 'RUNSPEC')
        # Ensure parser's dims state is set from RUNSPEC if available
        if 'DIMENS' in runspec_data and not parser.dims:
            parser.dims = tuple(map(int, runspec_data['DIMENS'][:3]))

        grid_data = parser._parse_section(sections.get('GRID', ''), 'GRID')
        props_data = parser._parse_section(sections.get('PROPS', ''), 'PROPS')
        solution_data = parser._parse_section(sections.get('SOLUTION', ''), 'SOLUTION')
        regions_data = parser._parse_section(sections.get('REGIONS', ''), 'REGIONS')
        summary_data = parser._parse_section(sections.get('SUMMARY', ''), 'SUMMARY')
        schedule_data = parser._parse_section(sections.get('SCHEDULE', ''), 'SCHEDULE')

        # 4. Combine and structure the data
        final_grid_data = {**grid_data, **props_data}
        
        # Extract PVT and Saturation tables into their own dictionaries
        pvt_tables = {}
        sat_tables = {}
        for kw in list(final_grid_data.keys()):
            if kw in PVT_KEYWORDS:
                pvt_tables[kw] = final_grid_data.pop(kw)
            elif kw in SAT_TABLE_KEYWORDS:
                sat_tables[kw] = final_grid_data.pop(kw)

        # Attach saturation data to props and other data to runspec for clarity
        final_grid_data.update(sat_tables)
        runspec_data['SOLUTION_DATA'] = solution_data
        runspec_data['SUMMARY_DATA'] = summary_data
        runspec_data['SCHEDULE_DATA'] = schedule_data
        
        if not any([runspec_data, final_grid_data, pvt_tables, regions_data]):
             raise EclipseParserError("Failed to parse any valid data from the ECLIPSE file.")

        logger.info(f"Successfully parsed ECLIPSE file: {path}")
        return ReservoirData(
            runspec=runspec_data,
            grid=final_grid_data,
            pvt_tables=pvt_tables,
            regions=regions_data
        )
    except Exception as e:
        logger.exception(f"A critical error occurred while parsing {path}")
        # Re-raise as a generic parser error to abstract away the specific internal exception
        raise EclipseParserError(f"Parsing failed for {path}: {e}") from e