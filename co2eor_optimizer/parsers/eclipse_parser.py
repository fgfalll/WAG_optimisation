import os
import sys
import numpy as np
import re
from pathlib import Path
import logging
from typing import Dict, Tuple, Optional, Any, List, Set

# Use the dataclass from the core module
from ..core import ReservoirData


# Setup a module-level logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define constants for Eclipse keywords
SECTION_KEYWORDS = ['RUNSPEC', 'GRID', 'PROPS', 'REGIONS', 'SOLUTION', 'SCHEDULE', 'SUMMARY']
DIMENS_KEYWORDS = ['DIMENS', 'SPECGRID']
IMPLICIT_GRID_KEYWORDS = ['DIMENS', 'SPECGRID', 'COORD', 'ZCORN']
PVT_KEYWORDS = ['PVTO', 'PVTG', 'PVDO', 'PVDG']
SAT_TABLE_KEYWORDS = ['SWFN', 'SGFN', 'SOF3', 'SWOF']
REGION_KEYWORDS = ['EQLNUM', 'FIPNUM', 'SATNUM']
ARRAY_MODIFICATION_KEYWORDS = ['COPY', 'ADD', 'MULTIPLY']


# Frequently used keywords as constants
DIMENS = "DIMENS"
SPECGRID = "SPECGRID"
COORD = "COORD"
ZCORN = "ZCORN"
PVTO = "PVTO" # Standardized key for oil PVT
PVTG = "PVTG" # Standardized key for gas PVT
PVDO = "PVDO" # Dead Oil alias
PVDG = "PVDG" # Dry Gas alias
EQUALS = "EQUALS"
COPY = "COPY"
ADD = "ADD"
MULTIPLY = "MULTIPLY"

# Precompiled regex patterns for performance
COMMENT_PATTERN = re.compile(r"\s*--.*")
INCLUDE_PATTERN = re.compile(
    r"^\s*INCLUDE\s+(?:\"([^\"]+)\"|'([^']+)'|(\S+))",
    re.IGNORECASE | re.MULTILINE
)

# Custom exceptions
class EclipseParserError(Exception):
    """Base class for Eclipse parser exceptions."""

class IncludeProcessingError(EclipseParserError):
    """Error during INCLUDE file processing."""

class SectionParsingError(EclipseParserError):
    """Error during section parsing."""

class PVTTableError(EclipseParserError):
    """Error during PVT table parsing."""

class DimensionError(EclipseParserError):
    """Error in grid dimension specification."""

class EclipseParser:
    """
    A comprehensive parser for ECLIPSE 100 data files.
    """
    def __init__(self):
        """Initializes the parser."""
        self.dims: Optional[Tuple[int, int, int]] = None
        self.logger = logging.getLogger(__name__)

    @staticmethod
    def _expand_eclipse_values(data_tokens: List[str], keyword: str) -> List[str]:
        """
        Expands 'N* value' and 'N*value' syntax in a list of tokens.
        """
        expanded_values = []
        i = 0
        while i < len(data_tokens):
            token = data_tokens[i]
            
            if '*' in token:
                try:
                    count_str, value_str = token.split('*', 1)

                    if value_str == '' and count_str.isdigit():
                        count = int(count_str)
                        if i + 1 < len(data_tokens):
                            value_to_repeat = data_tokens[i + 1]
                            # FIX: Handle '1* value' case to preserve token count.
                            if count == 1:
                                # Interpret '1* value' as two tokens: '1' and 'value'.
                                expanded_values.append(count_str)
                                expanded_values.append(value_to_repeat)
                                i += 2 # Consume both '1*' and 'value' tokens.
                            else:
                                # For N > 1, 'N* value' means repeat 'value' N times.
                                expanded_values.extend([value_to_repeat] * count)
                                i += 2 # Consume both 'N*' and 'value' tokens.
                            continue
                        else:
                            # This handles 'N*' at the end of a line, which usually implies
                            # repeating a default value. The original code used an empty string.
                            expanded_values.extend([''] * count)
                            i += 1
                            continue

                    elif count_str.isdigit():
                        count = int(count_str)
                        expanded_values.extend([value_str] * count)
                        i += 1
                        continue

                    else:
                        expanded_values.append(token)
                        i += 1

                except (ValueError, TypeError):
                    expanded_values.append(token)
                    i += 1
            else:
                expanded_values.append(token)
                i += 1
                
        return expanded_values

    def _parse_pvt_table(self, table_content: str, keyword: str) -> list:
        """Parses PVTO/PVTG tables into a structured hierarchical format."""
        try:
            cleaned = COMMENT_PATTERN.sub("", table_content)
            lines = cleaned.splitlines()
            table_data = []
            current_outer = None
            current_inner = []

            for line in lines:
                line = line.strip()
                if not line or line == '/':
                    continue

                raw_tokens = line.split()
                tokens = self._expand_eclipse_values(raw_tokens, keyword)

                values = []
                for token in tokens:
                    try:
                        values.append(float(token.lower().replace('d', 'e')))
                    except (ValueError, TypeError):
                        values.append(token)

                if not values: continue

                if isinstance(values[0], float):
                    if current_outer is not None:
                        current_outer[1] = current_inner
                        table_data.append(current_outer)
                    current_outer = [values[0], []]
                    current_inner = current_outer[1]
                    if len(values) > 1:
                        current_inner.append(values[1:])
                else:
                    if current_outer is None:
                        raise PVTTableError(f"Invalid {keyword} table: data found before reference value.")
                    current_inner.append(values)

            if current_outer is not None:
                current_outer[1] = current_inner
                table_data.append(current_outer)

            return table_data
        except Exception as e:
            self.logger.exception(f"Failed to parse {keyword} table")
            raise PVTTableError(f"PVT table parsing failed: {str(e)}") from e
    
    def _parse_generic_table(self, table_content: str, keyword: str) -> np.ndarray:
        """Parses generic multi-column tables like SWFN, SGFN, etc."""
        try:
            cleaned = COMMENT_PATTERN.sub("", table_content)
            lines = [line.strip() for line in cleaned.splitlines() if line.strip() and not line.strip() == '/']
            
            table_data = []
            for line in lines:
                raw_tokens = line.split()
                
                values = []
                for token in raw_tokens:
                    try:
                        values.append(float(token.lower().replace('d', 'e')))
                    except (ValueError, TypeError):
                        self.logger.warning(f"Non-numeric value '{token}' in {keyword} table line: '{line}'. Skipping token.")
                        continue
                
                if values:
                    table_data.append(values)
            
            if not table_data:
                self.logger.warning(f"No data found for keyword {keyword} after parsing.")
                return np.array([])

            return np.array(table_data, dtype=np.float64)
        except Exception as e:
            self.logger.exception(f"Failed to parse generic table for keyword {keyword}")
            raise SectionParsingError(f"Generic table parsing failed for {keyword}: {str(e)}") from e

    def _preprocess_special_blocks(self, section_content: str, existing_data: Dict[str, Any]) -> Tuple[Dict[str, Any], str]:
        """
        Finds and processes special multi-keyword blocks like EQUALS, COPY, MULTIPLY.
        Returns the data parsed from these blocks and the remaining section content.
        """
        parsed_data = existing_data.copy()
        remaining_content = section_content

        block_pattern = re.compile(r"^\s*(EQUALS|COPY|MULTIPLY)\s*\n(.*?)\n^\s*/", re.DOTALL | re.IGNORECASE | re.MULTILINE)
        
        for match in block_pattern.finditer(section_content):
            block_keyword = match.group(1).upper()
            block_content = match.group(2)
            
            remaining_content = remaining_content.replace(match.group(0), "")
            parsed_data[block_keyword] = True

            if not self.dims:
                self.logger.warning(f"Found {block_keyword} block but grid dimensions are not set. Skipping.")
                continue

            nx, ny, nz = self.dims

            for line in block_content.strip().splitlines():
                parts = COMMENT_PATTERN.sub("", line).strip().split()
                if not parts: continue
                parts = [p for p in parts if p != '/']

                if block_keyword == EQUALS:
                    if len(parts) < 8: continue
                    keyword, value_str = parts[0].upper(), parts[1]
                    try:
                        value = float(value_str)
                        i1, i2, j1, j2, k1, k2 = map(int, parts[2:8])
                        
                        if keyword not in parsed_data:
                            parsed_data[keyword] = np.zeros(nx * ny * nz)
                        
                        temp_grid = parsed_data[keyword].reshape((nz, ny, nx), order='F')
                        temp_grid[k1-1:k2, j1-1:j2, i1-1:i2] = value
                        parsed_data[keyword] = temp_grid.flatten(order='F')
                    except (ValueError, IndexError) as e:
                        self.logger.warning(f"Could not parse line in EQUALS block: '{line}'. Error: {e}")
                
                elif block_keyword in [COPY, MULTIPLY]:
                    source_kw, dest_kw = parts[0].upper(), parts[1].upper()
                    if source_kw not in parsed_data:
                        self.logger.warning(f"Source keyword '{source_kw}' for {block_keyword} not found. Skipping.")
                        continue
                    
                    source_array = parsed_data[source_kw]
                    if block_keyword == COPY:
                        parsed_data[dest_kw] = np.copy(source_array)
                    elif block_keyword == MULTIPLY:
                        if len(parts) >= 9:
                           value = float(parts[2])
                           i1, i2, j1, j2, k1, k2 = map(int, parts[3:9])
                           temp_grid = source_array.reshape((nz, ny, nx), order='F')
                           temp_grid[k1-1:k2, j1-1:j2, i1-1:i2] *= value
                           parsed_data[dest_kw] = temp_grid.flatten(order='F')
                        else:
                           value = float(parts[2])
                           parsed_data[dest_kw] = source_array * value
                            
        return parsed_data, remaining_content

    def _parse_keyword_data_blocks(self, section_content: str, section_name: str) -> Dict[str, Any]:
        """
        Parses keyword-data blocks from a section string using a robust splitting method.
        """
        parsed_data = {}
        keyword_pattern = re.compile(r"^\s*([A-Z_][A-Z0-9_]+)", re.M)
        matches = list(keyword_pattern.finditer(section_content))
        
        for i, match in enumerate(matches):
            keyword = match.group(1).upper()
            start_pos = match.start()
            end_pos = matches[i+1].start() if i + 1 < len(matches) else len(section_content)
            
            block_text = section_content[start_pos:end_pos]
            block_text = COMMENT_PATTERN.sub("", block_text)
            
            lines = block_text.strip().splitlines()
            if not lines:
                parsed_data[keyword] = np.nan
                continue
            
            data_tokens = []
            block_terminated = False
            for line in lines:
                if block_terminated: break
                parts = line.strip().split()
                if not parts: continue
                
                if parts[0].upper() == keyword:
                    parts = parts[1:]
                
                if len(parts) == 1 and parts[0] == '/':
                    block_terminated = True
                    break

                for part in parts:
                    if part.endswith('/'):
                        value = part[:-1]
                        if value: data_tokens.append(value)
                    else:
                        data_tokens.append(part)
                
            if keyword in PVT_KEYWORDS:
                table_content = "\n".join(lines[1:])
                parsed_data[keyword] = self._parse_pvt_table(table_content, keyword)
                continue

            if keyword in SAT_TABLE_KEYWORDS:
                table_content = "\n".join(lines[1:])
                parsed_data[keyword] = self._parse_generic_table(table_content, keyword)
                continue

            if not data_tokens:
                 parsed_data[keyword] = np.nan
                 continue

            expanded_tokens = self._expand_eclipse_values(data_tokens, keyword)
            numeric_values = []
            for token in expanded_tokens:
                try:
                    numeric_values.append(float(token.lower().replace('d', 'e')))
                except (ValueError, TypeError):
                    numeric_values.append(token.strip("'\""))

            if keyword in DIMENS_KEYWORDS:
                if len(numeric_values) < 3:
                    raise DimensionError(f"{keyword} requires at least 3 values, got {len(numeric_values)}")
                nx, ny, nz = map(int, numeric_values[:3])
                self.dims = (nx, ny, nz)
                self.logger.info(f"Parsed grid dimensions: {self.dims} from {keyword}")
                parsed_data[keyword] = list(self.dims)
                continue

            if section_name in ['GRID', 'PROPS', 'SOLUTION'] and self.dims:
                is_numeric_array = any(isinstance(v, (int, float)) for v in numeric_values)
                cell_property_keywords = ['TOPS', 'DXV', 'DYV', 'DZ', 'PORO', 'NTG', 'PERMX', 'PERMY', 'PERMZ', 'ACTNUM', 'PRESSURE', 'SWAT', 'SGAS']
                if is_numeric_array and keyword in cell_property_keywords:
                    expected_size = self.dims[0] * self.dims[1] * self.dims[2]

                    if len(numeric_values) == 1 and expected_size > 1:
                        numeric_values = [numeric_values[0]] * expected_size
                    elif len(numeric_values) < expected_size:
                        self.logger.warning(f"Keyword '{keyword}' has incomplete data: Expected {expected_size}, got {len(numeric_values)}. Padded with NaN.")
                        numeric_values.extend([np.nan] * (expected_size - len(numeric_values)))
                    elif len(numeric_values) > expected_size:
                        self.logger.warning(f"Keyword '{keyword}' has excess data: Expected {expected_size}, got {len(numeric_values)}. Truncated.")
                        numeric_values = numeric_values[:expected_size]
            
            try:
                # Attempt to convert to a numeric array first
                parsed_data[keyword] = np.array(numeric_values, dtype=np.float64)
            except ValueError:
                # If conversion fails, create an array of objects to handle mixed types
                parsed_data[keyword] = np.array(numeric_values, dtype=object)


        return parsed_data

    def _resolve_include_path(self, include_path: str, base_path: str) -> str:
        """Resolves an INCLUDE file path, prioritizing relative paths."""
        if os.path.isabs(include_path): return include_path
        base_dir = os.path.dirname(os.path.abspath(base_path))
        resolved_from_base = os.path.normpath(os.path.join(base_dir, include_path))
        if os.path.exists(resolved_from_base): return resolved_from_base
        resolved_from_cwd = os.path.normpath(os.path.join(os.getcwd(), include_path))
        if os.path.exists(resolved_from_cwd):
            self.logger.debug(f"Resolved '{include_path}' relative to CWD, not file location '{base_path}'.")
            return resolved_from_cwd
        return resolved_from_base

    def _process_includes(self, content: str, base_file_path: str, visited: Optional[Set[str]] = None) -> str:
        """Recursively processes INCLUDE statements."""
        if visited is None: visited = set()
        def replace_include(match):
            path = match.group(1) or match.group(2) or match.group(3)
            if not path: return ""
            resolved_path = self._resolve_include_path(path, base_file_path)
            abs_path = os.path.abspath(resolved_path)
            if abs_path in visited:
                self.logger.warning(f"Circular INCLUDE detected and skipped: {abs_path}")
                return ""
            if not os.path.exists(resolved_path):
                self.logger.warning(f"INCLUDE file not found: {resolved_path}")
                return ""
            visited.add(abs_path)
            try:
                with open(resolved_path, 'r', errors='ignore') as f:
                    included_content = f.read()
                return self._process_includes(included_content, resolved_path, visited)
            except IOError as e:
                self.logger.error(f"Error reading included file {resolved_path}: {str(e)}")
                return ""
        return INCLUDE_PATTERN.sub(replace_include, content)

    SECTION_HEADER_PATTERN = re.compile(
        r"^\s*(" + "|".join(re.escape(kw) for kw in SECTION_KEYWORDS) + r")\b", re.IGNORECASE)

    def _parse_by_sections(self, content: str) -> Dict[str, str]:
        """Parses content into sections based on major keywords."""
        sections = {}
        current_section = None
        current_lines = []
        lines = content.splitlines()
        for line in lines:
            line = COMMENT_PATTERN.sub("", line)
            if not line.strip():
                continue

            header_match = self.SECTION_HEADER_PATTERN.search(line)
            if header_match:
                if current_section: sections[current_section] = "\n".join(current_lines)
                current_section = header_match.group(1).upper()
                current_lines = [line]
                continue

            if current_section is None and any(kw in line.upper() for kw in IMPLICIT_GRID_KEYWORDS):
                current_section = 'GRID'
                self.logger.info("Implicit GRID section detected.")
            
            if current_section:
                current_lines.append(line)

        if current_section: sections[current_section] = "\n".join(current_lines)
        if not sections and content.strip():
            self.logger.info("No explicit sections found. Treating all content as RUNSPEC.")
            sections['RUNSPEC'] = content
        return sections

    REGION_KEYWORD_PATTERN = re.compile(
        r"^\s*(" + "|".join(re.escape(kw) for kw in REGION_KEYWORDS) + r")\b", re.IGNORECASE | re.MULTILINE)

    def _parse_regions(self, content: str) -> Dict[str, Any]:
        """Parses region keywords from the REGIONS section content."""
        regions_data = {}
        for match in self.REGION_KEYWORD_PATTERN.finditer(content):
            keyword = match.group(1).upper()
            start_pos = match.end()
            end_match = re.search(r'/\s*$', content[start_pos:], re.MULTILINE)
            if end_match: data_block = content[start_pos : start_pos + end_match.start()]
            else:
                next_kw_match = self.REGION_KEYWORD_PATTERN.search(content, start_pos)
                if next_kw_match: data_block = content[start_pos : next_kw_match.start()]
                else: data_block = content[start_pos:]
            cleaned_block = COMMENT_PATTERN.sub("", data_block).strip()
            tokens = self._expand_eclipse_values(cleaned_block.split(), keyword)
            try:
                numeric_values = [float(token.lower().replace('d', 'e')) for token in tokens]
                regions_data[keyword] = np.array(numeric_values)
            except ValueError:
                self.logger.warning(f"Could not convert all tokens to numbers for {keyword}. Storing as mixed list.")
                regions_data[keyword] = tokens
        return regions_data

def parse_eclipse(file_path: str) -> "ReservoirData":
    """
    Main function to parse an ECLIPSE 100 data file into a structured object.
    """
    parser = EclipseParser()
    try:
        logger.info(f"Starting Eclipse file parsing: {file_path}")
        if not Path(file_path).is_file(): raise IOError(f"File not found: {file_path}")
        with open(file_path, 'r', errors='ignore') as f: content = f.read()
        full_content = parser._process_includes(content, file_path)
        if not full_content.strip(): raise ValueError("Eclipse file is empty after INCLUDE processing.")
        sections = parser._parse_by_sections(full_content)
        
        runspec_data = parser._parse_keyword_data_blocks(sections.get('RUNSPEC', ''), 'RUNSPEC')
        if 'DIMENS' in runspec_data:
             parser.dims = tuple(runspec_data['DIMENS'])

        grid_content = sections.get('GRID', '')
        props_content = sections.get('PROPS', '')
        
        processed_data, remaining_grid_content = parser._preprocess_special_blocks(grid_content, {})
        processed_data, remaining_props_content = parser._preprocess_special_blocks(props_content, processed_data)
        
        remaining_grid_data = parser._parse_keyword_data_blocks(remaining_grid_content, 'GRID')
        remaining_props_data = parser._parse_keyword_data_blocks(remaining_props_content, 'PROPS')
        solution_data = parser._parse_keyword_data_blocks(sections.get('SOLUTION', ''), 'SOLUTION')
        summary_data = parser._parse_keyword_data_blocks(sections.get('SUMMARY', ''), 'SUMMARY')
        schedule_data = parser._parse_keyword_data_blocks(sections.get('SCHEDULE', ''), 'SCHEDULE')
        regions_data = parser._parse_regions(sections.get('REGIONS', ''))

        grid_data = {**remaining_grid_data, **processed_data}
        props_data = {**remaining_props_data}
        
        pvt_data = {}
        if PVTO in props_data: pvt_data[PVTO] = props_data.pop(PVTO)
        if PVDO in props_data: pvt_data[PVTO] = props_data.pop(PVDO)
        if PVTG in props_data: pvt_data[PVTG] = props_data.pop(PVTG)
        if PVDG in props_data: pvt_data[PVTG] = props_data.pop(PVDG)
        
        grid_data.update(props_data)

        if not any([runspec_data, grid_data, pvt_data, regions_data, solution_data, summary_data, schedule_data]):
            raise ValueError("Failed to parse any valid data from the Eclipse file.")
        
        runspec_data['SOLUTION_DATA'] = solution_data
        runspec_data['SUMMARY_DATA'] = summary_data
        runspec_data['SCHEDULE_DATA'] = schedule_data
        
        logger.info(f"Successfully parsed Eclipse file: {file_path}")
        return ReservoirData(
            runspec=runspec_data,
            grid=grid_data,
            pvt_tables=pvt_data,
            regions=regions_data
        )
    except Exception as e:
        logger.exception(f"Critical error parsing Eclipse file: {file_path}")
        raise EclipseParserError(f"Eclipse parsing failed: {str(e)}") from e