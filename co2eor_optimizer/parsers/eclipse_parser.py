import re
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional, Any, List, Set, Union

import numpy as np

from co2eor_optimizer.core.data_models import ReservoirData, WellData
from .base_parser import BaseParser
from .validation import validate_reservoir_data

# --- Constants for Eclipse Keywords ---
SECTION_KEYWORDS = ['RUNSPEC', 'GRID', 'PROPS', 'REGIONS', 'SOLUTION', 'SCHEDULE', 'SUMMARY']
DIMENS_KEYWORDS = ['DIMENS', 'SPECGRID']
PVT_KEYWORDS = {'PVTO', 'PVDO', 'PVTG', 'PVDG', 'PVTW'}
SAT_TABLE_KEYWORDS = {'SWFN', 'SGFN', 'SOF3', 'SWOF'}
ARRAY_MODIFICATION_KEYWORDS = ['COPY', 'ADD', 'MULTIPLY']
TERMINATOR = '/'

# --- Precompiled Regex Patterns for Performance ---
COMMENT_PATTERN = re.compile(r"--.*")
INCLUDE_PATTERN = re.compile(r"^\s*INCLUDE\s*['\"]([^'\"]+)['\"]", re.IGNORECASE | re.MULTILINE)
KEYWORD_PATTERN = re.compile(r"^\s*([A-Z_][A-Z0-9_]*)\s*", re.IGNORECASE | re.MULTILINE)

# --- Custom Exceptions for Clearer Error Reporting ---
class EclipseParserError(Exception): pass
class IncludeProcessingError(EclipseParserError): pass
class SectionParsingError(EclipseParserError): pass
class DimensionError(EclipseParserError): pass


class EclipseParser(BaseParser):
    def __init__(self):
        super().__init__()
        self.dims: Optional[Tuple[int, int, int]] = None
        self.reservoir_data: Optional[ReservoirData] = None

    def reset_state(self):
        super().reset_state()
        self.dims = None
        self.reservoir_data = None

    def _log_parsed_item(self, section: str, keyword: str, value: Any):
        log_msg = f"[{section}] Parsed Keyword: '{keyword}' | Type: {type(value)}"
        if isinstance(value, np.ndarray):
            log_msg += f" | Shape: {value.shape} | Dtype: {value.dtype}"
        elif isinstance(value, list):
            log_msg += f" | Length: {len(value)}"
        self.logger.debug(log_msg)

    @staticmethod
    def _tokenize_line(line: str) -> List[str]:
        line_no_commas = line.replace(',', ' ')
        tokens = re.findall(r"\'[^\']+\'|\"[^\"]+\"|\S+", line_no_commas)
        return [token.strip("'\"").strip() for token in tokens]

    @staticmethod
    def _expand_eclipse_values(data_tokens: List[str]) -> List[str]:
        expanded_values = []
        for token in data_tokens:
            if '*' not in token:
                expanded_values.append(token)
                continue
            try:
                count_str, value_str = token.split('*', 1)
                if count_str.isdigit():
                    expanded_values.extend([value_str] * int(count_str))
                else: expanded_values.append(token)
            except ValueError: expanded_values.append(token)
        return expanded_values

    def _parse_pvt_or_sat_table(self, content: str, keyword: str) -> np.ndarray:
        lines = [line.strip() for line in content.splitlines() if line.strip() and not line.strip().startswith(TERMINATOR)]
        table_data = []
        for line in lines:
            cleaned_line = COMMENT_PATTERN.sub("", line).strip().rstrip(TERMINATOR).strip()
            if not cleaned_line: continue
            raw_tokens = self._tokenize_line(cleaned_line)
            expanded_tokens = self._expand_eclipse_values(raw_tokens)
            try:
                values = [float(token.lower().replace('d', 'e')) for token in expanded_tokens if token]
                if values: table_data.append(values)
            except ValueError:
                 self.logger.warning(f"Skipping non-numeric line in {keyword} table: '{line}'")
        if not table_data: return np.array([])
        max_len = max(len(row) for row in table_data) if table_data else 0
        for row in table_data: row.extend([np.nan] * (max_len - len(row)))
        return np.array(table_data, dtype=float)

    def _apply_array_modifications(self, data: Dict[str, Any], keyword: str, block_content: str):
        if not self.dims: self.logger.warning(f"Found '{keyword}' but dims not set. Skipping."); return
        nx, ny, nz = self.dims
        lines = block_content.strip().splitlines()
        for line in lines:
            line_content = COMMENT_PATTERN.sub("", line).strip().rstrip(TERMINATOR).strip()
            if not line_content: continue
            parts = self._tokenize_line(line_content)
            if not parts: continue
            try:
                if keyword == 'COPY':
                    if len(parts) < 2: continue
                    source_kw, dest_kw = parts[0].upper(), parts[1].upper()
                    if source_kw not in data: self.logger.warning(f"Source '{source_kw}' for COPY not found."); continue
                    data[dest_kw] = np.copy(data[source_kw])
                elif keyword in ('ADD', 'MULTIPLY'):
                    if len(parts) < 2: continue
                    target_kw = parts[0].upper()
                    if target_kw not in data: self.logger.warning(f"Target keyword '{target_kw}' for {keyword} not found."); continue
                    value = float(parts[1])
                    box = tuple(map(int, parts[2:8])) if len(parts) >= 8 else (1, nx, 1, ny, 1, nz)
                    grid_view = data[target_kw].reshape((nz, ny, nx), order='F')
                    region_slice = (slice(box[4]-1, box[5]), slice(box[2]-1, box[3]), slice(box[0]-1, box[1]))
                    if keyword == 'ADD': grid_view[region_slice] += value
                    elif keyword == 'MULTIPLY': grid_view[region_slice] *= value
            except (ValueError, IndexError) as e:
                self.logger.warning(f"Could not parse line in {keyword} block: '{line}'. Error: {e}")

    def _parse_keyword_value_block(self, block_content: str) -> Dict[str, Any]:
        parsed_data = {}
        if not self.dims: return {}
        nx, ny, nz = self.dims
        total_cells = nx * ny * nz
        lines = block_content.strip().splitlines()
        for line in lines:
            line_content = COMMENT_PATTERN.sub("", line).strip().rstrip(TERMINATOR).strip()
            if not line_content: continue
            tokens = self._tokenize_line(line_content)
            if not tokens: continue
            keyword = tokens[0].upper()
            values = tokens[1:]
            try: float(values[0])
            except (ValueError, IndexError): continue
            
            box = (1, nx, 1, ny, 1, nz)
            box_was_found = False
            if len(values) >= 7:
                try:
                    box_vals = [int(v) for v in values[-6:]]
                    if all(isinstance(v, int) for v in box_vals):
                        box = tuple(box_vals); values = values[:-6]; box_was_found = True
                except (ValueError, IndexError): pass
            
            if not values: continue
            const_val = float(values[0])

            # --- THE ONLY FIX THAT MATTERS ---
            # If no box is explicitly given, check for 1D keywords.
            if not box_was_found:
                if keyword in ('DX', 'DXV'):
                    parsed_data[keyword] = np.full(nx, const_val, dtype=np.float64)
                    continue
                elif keyword in ('DY', 'DYV'):
                    parsed_data[keyword] = np.full(ny, const_val, dtype=np.float64)
                    continue
            
            # For all other keywords, or if a box was found, use box logic.
            if keyword not in parsed_data:
                parsed_data[keyword] = np.full(total_cells, np.nan, dtype=np.float64)
            
            grid_view = parsed_data[keyword].reshape((nz, ny, nx), order='F')
            i1, i2, j1, j2, k1, k2 = box
            grid_view[k1-1:k2, j1-1:j2, i1-1:i2] = const_val
        return parsed_data

    def _parse_section(self, section_content: str, section_name: str, current_data: Dict[str, Any]):
        content_ptr = 0
        while content_ptr < len(section_content):
            match = KEYWORD_PATTERN.search(section_content, content_ptr)
            if not match: break
            keyword = match.group(1).upper()
            if not keyword: content_ptr = match.end(); continue
            block_start = match.end()
            next_match = KEYWORD_PATTERN.search(section_content, block_start)
            block_end = next_match.start() if next_match else len(section_content)
            block_content = section_content[block_start:block_end]
            content_ptr = block_end
            
            if keyword in ARRAY_MODIFICATION_KEYWORDS:
                self._apply_array_modifications(current_data, keyword, block_content)
                continue
            if keyword == 'EQUALS':
                block_data = self._parse_keyword_value_block(block_content)
                current_data.update(block_data)
                continue
            if keyword in PVT_KEYWORDS or keyword in SAT_TABLE_KEYWORDS:
                current_data[keyword] = self._parse_pvt_or_sat_table(block_content, keyword)
                continue
                
            data_tokens = []
            lines = block_content.strip().splitlines()
            for line in lines:
                clean_line = COMMENT_PATTERN.sub("", line).strip().rstrip(TERMINATOR).strip()
                if clean_line: data_tokens.extend(self._tokenize_line(clean_line))
            
            if not data_tokens:
                current_data[keyword] = True
                continue
                
            expanded_tokens = self._expand_eclipse_values(data_tokens)
            try: numeric_values = [float(token.lower().replace('d', 'e')) for token in expanded_tokens]
            except (ValueError, TypeError): numeric_values = expanded_tokens
            
            if keyword in DIMENS_KEYWORDS:
                if len(numeric_values) < 3: raise DimensionError(f"{keyword} needs >= 3 values")
                if self.dims is None:
                    self.dims = tuple(map(int, numeric_values[:3]))
                    self.logger.info(f"SUCCESS: Set grid dimensions to {self.dims} from {section_name} section.")
                current_data[keyword] = list(map(int, numeric_values[:3]))
                continue
            
            try: current_data[keyword] = np.array(numeric_values, dtype=np.float64)
            except (ValueError, TypeError): current_data[keyword] = np.array(numeric_values, dtype=object)
            self._log_parsed_item(section_name, keyword, current_data[keyword])

    def _split_into_sections(self, content: str) -> Dict[str, str]:
        sections = {}
        pattern = re.compile(r'^\s*(' + '|'.join(SECTION_KEYWORDS) + r')\b', re.IGNORECASE | re.MULTILINE)
        matches = list(pattern.finditer(content))
        if not matches: sections['_SINGLE_BLOCK_'] = content; return sections
        for i, current_match in enumerate(matches):
            keyword = current_match.group(1).upper()
            start_pos = current_match.end()
            end_pos = matches[i+1].start() if i + 1 < len(matches) else len(content)
            sections[keyword] = content[start_pos:end_pos]
        return sections

    def _resolve_include_path(self, include_path: str, base_path: str) -> Path:
        # ... (unchanged)
        resolved_path = Path(base_path).parent / include_path
        if not resolved_path.exists(): resolved_path = Path.cwd() / include_path
        return resolved_path.resolve()

    def _process_includes(self, content: str, base_file_path: str, visited: Optional[Set[Path]] = None) -> str:
        # ... (unchanged)
        if visited is None: visited = set()
        current_abs_path = Path(base_file_path).resolve()
        visited.add(current_abs_path)
        def replace_include(match):
            resolved_path = self._resolve_include_path(match.group(1), base_file_path)
            if resolved_path in visited: self.logger.warning(f"Circular INCLUDE skipped: {resolved_path}"); return ""
            if not resolved_path.exists(): self.logger.error(f"INCLUDE file not found: {resolved_path}"); return ""
            try:
                with resolved_path.open('r', errors='ignore') as f:
                    return self._process_includes(f.read(), str(resolved_path), visited)
            except IOError as e: raise IncludeProcessingError(f"Error reading include {resolved_path}: {e}") from e
        return INCLUDE_PATTERN.sub(replace_include, content)

    def parse(self, file_path: Union[str, Path]) -> bool:
        self.reset_state()
        self._log_parse_start(file_path)
        try:
            path = Path(file_path)
            with path.open('r', errors='ignore') as f: initial_content = f.read()
            full_content = self._process_includes(initial_content, str(path))

            sections = self._split_into_sections(full_content)
            all_data = {}
            
            section_order = ['RUNSPEC', 'GRID', 'PROPS', 'REGIONS', 'SOLUTION', 'SCHEDULE', 'SUMMARY']
            for section_name in section_order:
                if section_name in sections:
                    self._parse_section(sections.pop(section_name), section_name, all_data)
            
            if not self.dims: raise DimensionError("DIMENS keyword not found.")
            
            runspec_data = {k:v for k,v in all_data.items() if not isinstance(v, np.ndarray) or k in DIMENS_KEYWORDS}
            grid_data = {k:v for k,v in all_data.items() if isinstance(v, np.ndarray) and k not in runspec_data}
            pvt_tables = {kw: grid_data.pop(kw) for kw in list(grid_data.keys()) if kw in PVT_KEYWORDS or kw in SAT_TABLE_KEYWORDS}
            
            runspec_data['DIMENSIONS'] = list(self.dims)
            self.reservoir_data = ReservoirData(runspec=runspec_data, grid=grid_data, pvt_tables=pvt_tables, regions={}, schedule={}, summary={})
            
            if not self.validate():
                self.logger.error(f"Parse failed due to validation errors.")
                return False

            success = True
        except (FileNotFoundError, EclipseParserError, DimensionError, Exception) as e:
            self.logger.exception(f"Critical error parsing {file_path}: {e}")
            self.validation_errors.append(str(e))
            success = False
        finally:
            self._log_parse_end(success)
        return success

    def to_reservoir_data(self) -> Optional[ReservoirData]:
        return self.reservoir_data

    def validate(self) -> bool:
        if self.reservoir_data is None: self.validation_errors.append("No data to validate."); return False
        self.validation_errors = validate_reservoir_data(self.reservoir_data)
        if self.validation_errors:
            for error in self.validation_errors: self.logger.error(f"Validation error: {error}")
            return False
        return True

    def to_well_data(self) -> Optional[WellData]:
        return None