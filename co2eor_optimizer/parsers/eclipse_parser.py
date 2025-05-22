from typing import Dict, Tuple, Optional, Any, List # Added List
import numpy as np
import re
from pathlib import Path
import logging
from ..core import ReservoirData

logger = logging.getLogger(__name__) # Use module-specific logger

# Helper function to expand N*value syntax
def _expand_eclipse_values(data_tokens: List[str], keyword: str) -> List[str]:
    expanded_values = []
    for token in data_tokens:
        if '*' in token:
            try:
                count_str, value_str = token.split('*', 1)
                count = int(count_str)
                # Ensure value_str is not empty, which can happen with "N*"
                if not value_str: 
                    logging.warning(f"Potentially malformed N*value token '{token}' for keyword {keyword}. Treating as single value.")
                    expanded_values.append(value_str if value_str else token) # if value_str is empty, keep token
                    continue

                expanded_values.extend([value_str] * count)
            except ValueError:
                logging.warning(f"Failed to parse N*value token '{token}' for keyword {keyword}. Storing as is.")
                expanded_values.append(token)
        else:
            expanded_values.append(token)
    return expanded_values

def _parse_keyword_data_blocks(section_content: str) -> Dict[str, Any]:
    """
    Parses keyword-data blocks from a section of an Eclipse file.
    A keyword block is generally:
    KEYWORD [optional_params_on_same_line]
      data line 1
      data line 2
      ...
    /
    
    Special handling for grid modification keywords:
    - COPY, ADD, MULTIPLY, BOX, EQUALS
    """
    data_blocks = {}
    current_keyword = None
    current_data = []
    
    for line in section_content.splitlines():
        line = line.strip()
        if not line or line.startswith('--'):
            continue
            
        # Handle grid modification operators
        if line.upper() in ('COPY', 'ADD', 'MULTIPLY', 'BOX', 'EQUALS'):
            if current_keyword:
                data_blocks[current_keyword] = current_data
            current_keyword = line.upper()
            current_data = []
        elif line == '/':
            if current_keyword:
                data_blocks[current_keyword] = current_data
            current_keyword = None
            current_data = []
        elif current_keyword:
            current_data.append(line)
            
    """Parses an Eclipse data file section into structured grid and property data.

    Handles:
    - Keyword detection (DIMENS, SPECGRID, COORD, ZCORN, etc.)
    - N*value expansion (e.g., 4*0.25)
    - Dimension validation
    - Critical grid data validation (COORD, ZCORN)
    - Numeric conversion with error handling

    Args:
        section_content: String containing the Eclipse data section to parse
        parsed_data: Dictionary to store parsed results (modified in-place)
        logger: Logger instance for diagnostic messages

    Returns:
        Dict[str, Any]: Parsed data structure with:
            - DIMENS: Tuple of (nx, ny, nz) dimensions
            - SPECGRID_DIMS: Tuple of (nx, ny, nz) from SPECGRID
            - COORD: Numpy array of grid coordinates
            - ZCORN: Numpy array of z-coordinates
            - Other property arrays (PORO, PERM*, etc.)
    """
    current_dims: Optional[Tuple[int, int, int]] = None
    parsed_data = {}
    # Enhanced pattern to handle:
    # 1. Keywords with inline parameters
    # 2. Multi-line data blocks
    # 3. Comments after keywords
    # 4. Various termination styles
    block_pattern = re.compile(
        r"^\s*([A-Z_][A-Z0-9_]*)\s*"          # Group 1: Keyword
        r"([^\n/]*?)\s*"                       # Group 2: Optional same-line params (non-greedy)
        r"(?:--[^\n]*)?\n"                     # Optional comment after keyword line
        r"([\s\S]*?)"                          # Group 3: Data block (non-greedy)
        r"(?:(?<!\S)^\s*/\s*$|(?=\n\s*[A-Z_]))", # Terminator: / on own line OR next keyword
        re.MULTILINE | re.IGNORECASE
    )

    for match in block_pattern.finditer(section_content):
        keyword = match.group(1).upper()
        params_on_keyword_line_str = match.group(2).strip()
        data_block_str = match.group(3).strip()
        
        all_tokens_for_keyword = []

        if keyword in ['DIMENS', 'SPECGRID']:
            # These keywords define dimensions using parameters on the keyword line itself.
            params = params_on_keyword_line_str.split()
            try:
                if keyword == 'DIMENS' and len(params) >= 3:
                    nx, ny, nz = map(int, params[:3])
                    current_dims = (nx, ny, nz)
                    parsed_data['DIMENS'] = current_dims
                    logger.info(f"Parsed {keyword}: NX={nx}, NY={ny}, NZ={nz}")
                elif keyword == 'SPECGRID' and len(params) >= 3:
                    nx, ny, nz = map(int, params[:3])
                    current_dims = (nx, ny, nz)
                    # Store as DIMENS for unified access, and SPECGRID_DIMS for specificity
                    parsed_data['DIMENS'] = current_dims
                    parsed_data['SPECGRID_DIMS'] = current_dims
                    if len(params) > 3:
                        parsed_data['SPECGRID_LGR'] = params[3]
                    if len(params) > 4:
                        try:
                            parsed_data['SPECGRID_PILLAR_FLAG'] = int(params[4])
                        except ValueError:
                            logger.warning(f"Could not parse pillar flag for SPECGRID: {params[4]}")
                    logger.info(f"Parsed {keyword}: NX={nx}, NY={ny}, NZ={nz}")
                # If data_block_str is not empty for DIMENS/SPECGRID, it's unusual but parse it if it exists.
                if data_block_str:
                    all_tokens_for_keyword.extend(data_block_str.split())

            except ValueError:
                logger.warning(f"Could not parse parameters for {keyword}: {params_on_keyword_line_str}")

        
        if not (keyword in ['DIMENS', 'SPECGRID'] and not data_block_str): # Avoid re-processing if only keyword line params
            if params_on_keyword_line_str and not data_block_str and keyword not in ['DIMENS', 'SPECGRID']:
                 pass # Complex case, may need more specific handling or regex adjustment

            if data_block_str:
                all_tokens_for_keyword.extend(data_block_str.split())
            elif params_on_keyword_line_str and keyword not in ['DIMENS', 'SPECGRID']: # keyword val1 val2 /
                all_tokens_for_keyword.extend(params_on_keyword_line_str.split())


        if not all_tokens_for_keyword:
            if keyword not in ['DIMENS', 'SPECGRID']: # DIMENS/SPECGRID might have no further data block.
                 logger.debug(f"Keyword {keyword} has no associated data tokens after initial parsing.")
            continue

        expanded_str_values = _expand_eclipse_values(all_tokens_for_keyword, keyword)
        
        numeric_values = []
        for s_val in expanded_str_values:
            # Special handling for critical grid data
            if keyword in ['COORD', 'ZCORN']:
                try:
                    val = float(s_val)
                    if np.isinf(val):
                        raise ValueError("Infinite value")
                    numeric_values.append(val)
                except ValueError as e:
                    logger.error(
                        f"Critical: Failed to parse {keyword} value '{s_val}': {str(e)}. "
                        "Grid geometry will be corrupted."
                    )
                    numeric_values.append(np.nan)
            else:
                # More lenient handling for properties
                try:
                    numeric_values.append(float(s_val))
                except ValueError:
                    logger.warning(f"Could not convert value '{s_val}' to float for {keyword}")
                    numeric_values.append(np.nan)
        
        if numeric_values:
            array_data = np.array(numeric_values)
            
            # Validate array sizes if dimensions are known
            if current_dims:
                nx, ny, nz = current_dims
                total_cells = nx * ny * nz
                expected_size = 0
                
                # Define expected sizes for different keyword types
                if keyword == 'ZCORN':
                    expected_size = total_cells * 8
                elif keyword == 'COORD':
                    expected_size = (nx + 1) * (ny + 1) * 6 # For pillar grids
                elif keyword in ['ACTNUM', 'PORO', 'NTG', 'SWAT', 'PRESSURE'] or keyword.startswith(("PERM", "TRAN")):
                    expected_size = total_cells
                
                if expected_size > 0:
                    actual_size = len(array_data)
                    if actual_size != expected_size:
                        msg = (f"{keyword} has {actual_size} values, "
                              f"expected {expected_size} for {nx}x{ny}x{nz} grid")
                              
                        if keyword in ['COORD', 'ZCORN']:
                            # Critical geometry must match exactly
                            logger.error(f"Critical: {msg}. Grid will be corrupted.")
                            if actual_size < expected_size:
                                # Pad critical geometry with NaN
                                padded = np.full(expected_size, np.nan)
                                padded[:actual_size] = array_data
                                array_data = padded
                                logger.warning(f"Padded {keyword} with NaN to expected size")
                            else:
                                # Truncate to expected size
                                array_data = array_data[:expected_size]
                                logger.warning(f"Truncated {keyword} to expected size")
                        else:
                            # Properties can be more flexible
                            if actual_size < expected_size:
                                logger.warning(f"{msg}. Repeating last value.")
                                array_data = np.resize(array_data, expected_size)
                            else:
                                logger.warning(f"{msg}. Using first {expected_size} values.")
                                array_data = array_data[:expected_size]
                    elif len(array_data) > expected_size:
                        logger.warning(
                            f"{keyword} has {len(array_data)} values, "
                            f"expected {expected_size}. Truncating to expected size."
                        )
                        array_data = array_data[:expected_size]
            
            # Store the parsed array. If keyword appears multiple times, last one wins (simplification).
            parsed_data[keyword] = array_data
            logger.info(f"Parsed {keyword} with {len(array_data)} values.")
        elif keyword not in ['DIMENS', 'SPECGRID']: # DIMENS/SPECGRID might not have numeric_values if handled by line params
             logger.warning(f"No parseable numeric data ultimately found for keyword {keyword}.")

    # Enhanced dimension fallback logic
    if 'DIMENS' not in parsed_data and 'SPECGRID_DIMS' not in parsed_data:
        # Try more flexible patterns that account for various Eclipse file formats
        dimens_pattern = r'^\s*DIMENS\s+(\d+)\s+(\d+)\s+(\d+)(?:\s+[^\n]*)?\s*(?:/\s*)?$'
        specgrid_pattern = r'^\s*SPECGRID\s+(\d+)\s+(\d+)\s+(\d+)(?:\s+[^\n]*)?\s*(?:/\s*)?$'
        
        for pattern, key in [(dimens_pattern, 'DIMENS'), (specgrid_pattern, 'SPECGRID_DIMS')]:
            match = re.search(pattern, section_content, re.IGNORECASE | re.MULTILINE)
            if match:
                try:
                    nx, ny, nz = map(int, match.groups()[:3])
                    parsed_data['DIMENS'] = (nx, ny, nz)
                    if key == 'SPECGRID_DIMS':
                        parsed_data[key] = (nx, ny, nz)
                    logger.warning(
                        f"Fallback parsed dimensions from {key}: "
                        f"NX={nx}, NY={ny}, NZ={nz}. "
                        "Consider adding explicit DIMENS/SPECGRID to input file."
                    )
                    break
                except (ValueError, IndexError) as e:
                    logger.error(f"Failed to parse fallback dimensions: {str(e)}")
        
        if 'DIMENS' not in parsed_data:
            logger.error(
                "Could not determine grid dimensions from input file. "
                "Grid properties may be incorrectly interpreted."
            )
    return parsed_data

def _parse_runspec_section(content: str) -> Dict[str, Any]:
    """Parse RUNSPEC section of Eclipse file."""
    runspec_data: Dict[str, Any] = {}
    section_match = re.search(
        r'\bRUNSPEC\b\s*\n([\s\S]*?)(?=\n\s*/\s*(?:\n|\Z|\b(?:GRID|PROPS|REGIONS|SOLUTION|SCHEDULE)\b))',
        content,
        re.IGNORECASE | re.DOTALL
    )
    
    if not section_match:
        logger.warning("No RUNSPEC section found in Eclipse file")
        return runspec_data
        
    section_content = section_match.group(1)
    
    # Parse standard keywords
    runspec_data.update(_parse_keyword_data_blocks(section_content))
    
    # Handle special RUNSPEC cases
    title_match = re.search(r'^TITLE\s*\n(.*?)(?=\n\s*/\s*$)', section_content, re.IGNORECASE | re.DOTALL)
    if title_match:
        runspec_data['TITLE'] = title_match.group(1).strip()
    
    return runspec_data

def _parse_faults_section(content: str) -> Dict[str, Any]:
    """Parse FAULTS section of Eclipse file."""
    faults_data: Dict[str, Any] = {}
    section_match = re.search(
        r'\bFAULTS\b\s*\n([\s\S]*?)(?=\n\s*/\s*(?:\n|\Z|\bEND\b))',
        content,
        re.IGNORECASE | re.DOTALL
    )
    
    if not section_match:
        logger.debug("No FAULTS section found in Eclipse file")
        return faults_data
        
    section_content = section_match.group(1)
    
    # Parse fault definitions
    fault_defs = []
    multflt_defs = []
    nnc_defs = []
    
    for line in section_content.splitlines():
        line = line.strip()
        if not line or line.startswith('--'):
            continue
            
        # Handle FAULT keyword
        if line.upper().startswith('FAULT'):
            parts = line.split()
            if len(parts) < 2:
                logger.warning(f"Invalid FAULT definition: {line}")
                continue
                
            try:
                try:
                    fault = {
                        'name': parts[1],
                        'faces': parts[2] if len(parts) > 2 else 'ALL',
                        'mult': float(parts[3]) if len(parts) > 3 else 1.0
                    }
                except (ValueError, IndexError) as e:
                    logger.warning(f"Invalid FAULT definition format: {line} - {str(e)}")
                    continue
            except (ValueError, IndexError) as e:
                logger.warning(f"Invalid FAULT definition format: {line} - {str(e)}")
                continue
            fault_defs.append(fault)
            
        # Handle MULTFLT keyword
        elif line.upper().startswith('MULTFLT'):
            parts = line.split()
            if len(parts) < 2:
                logger.warning(f"Invalid MULTFLT definition: {line}")
                continue
                
            multflt = {
                'name': parts[1],
                'mult': float(parts[2]) if len(parts) > 2 else 1.0
            }
            multflt_defs.append(multflt)
            
        # Handle NNC/EDITNNC keywords
        elif line.upper().startswith(('NNC', 'EDITNNC')):
            parts = line.split()
            if len(parts) < 7:
                logger.warning(f"Invalid NNC definition: {line}")
                continue
                
            nnc = {
                'type': parts[0].upper(),
                'i1': int(parts[1]),
                'j1': int(parts[2]),
                'k1': int(parts[3]),
                'i2': int(parts[4]),
                'j2': int(parts[5]),
                'k2': int(parts[6]),
                'trans': float(parts[7]) if len(parts) > 7 else None
            }
            nnc_defs.append(nnc)
    
    faults_data['faults'] = fault_defs
    faults_data['multflt'] = multflt_defs
    faults_data['nnc'] = nnc_defs
    
    # TODO: Review placement of fault processing logic
    faults_data['FAULTS'] = fault_defs
    return faults_data

def _parse_grid_section(content: str) -> Dict[str, np.ndarray]:
    """Parse GRID section of Eclipse file using keyword_data_blocks logic."""
    grid_data: Dict[str, np.ndarray] = {}
    grid_section_match = re.search(
        r'\bGRID\b\s*\n([\s\S]*?)(?=\n\s*/\s*(?:\n|\Z|\b(?:REGIONS|PROPS|SOLUTION|END)\b))',
        content,
        re.IGNORECASE | re.DOTALL
    )
    
    if not grid_section_match:
        logger.warning("No GRID section found in Eclipse file or section is improperly terminated.")
        # Attempt to parse the entire content if GRID keyword is missing but keywords like DIMENS exist
        # This is a fallback for GRDECL files that might not have an explicit GRID header.
        if re.search(r'^\s*(DIMENS|SPECGRID)\s*', content, re.IGNORECASE | re.MULTILINE):
            logger.info("GRID keyword not found, but DIMENS/SPECGRID detected. Parsing entire content for grid keywords.")
            grid_data.update(_parse_keyword_data_blocks(content))
        return grid_data
        
    grid_content = grid_section_match.group(1)
    grid_data.update(_parse_keyword_data_blocks(grid_content))
    
    return grid_data
    
def _resolve_include_path(include_path: str, base_path: str) -> str:
    """Resolve relative include paths to absolute paths"""
    import os
    if os.path.isabs(include_path):
        return include_path
    return os.path.normpath(os.path.join(os.path.dirname(base_path), include_path))

def _process_includes(content: str, base_path: str) -> str:
    """Process INCLUDE statements recursively"""
    import os
    include_pattern = re.compile(r'^\s*INCLUDE\s+[\'"]?(.*?)[\'"]?\s*$', re.IGNORECASE | re.MULTILINE)
    
    def replace_include(match):
        include_file = match.group(1).strip()
        resolved_path = _resolve_include_path(include_file, base_path)
        if not os.path.exists(resolved_path):
            logger.warning(f"INCLUDE file not found: {resolved_path}")
            return ""
        
        with open(resolved_path, 'r') as f:
            included_content = f.read()
            return _process_includes(included_content, os.path.dirname(resolved_path))
    
    return include_pattern.sub(replace_include, content)

def parse_eclipse(file_path: str) -> ReservoirData:
    """
    Parse ECLIPSE 100 format into ReservoirData.
    Focuses on GRID section for geometry and properties, and PROPS for PVT.
    Handles INCLUDE statements recursively.
    
    Args:
        file_path: Path to Eclipse input file
        
    Returns:
        ReservoirData object containing parsed grid, properties and PVT data
        
    Raises:
        IOError: If file cannot be read
        ValueError: For invalid or empty files
        RuntimeError: For critical parsing errors
    """
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Process INCLUDE statements first
        content = _process_includes(content, file_path)
    except Exception as e:
        logger.error(f"Failed to read Eclipse file: {file_path}. Error: {e}")
        raise IOError(f"Eclipse file read error: {str(e)}")

    if not content.strip():
        logger.error(f"Empty Eclipse file: {file_path}")
        raise ValueError("Empty Eclipse file")
            
    # Essential keywords for validation
    essential_keywords = {
        'sections': ['RUNSPEC', 'GRID', 'PROPS', 'REGIONS', 'SOLUTION', 'SCHEDULE'],
        'grid': ['DIMENS', 'SPECGRID', 'COORD', 'ZCORN'],
        'properties': ['PORO', 'PERMX', 'PERMY', 'PERMZ', 'NTG'],
        'runspec': ['TITLE', 'DIMENS', 'OIL', 'GAS', 'WATER', 'DISGAS', 'VAPOIL']
    }
    
    content_upper = content.upper()
    missing = []
    
    # Check for at least one section
    if not any(section in content_upper for section in essential_keywords['sections']):
        missing.append("sections")
        
    # Check for grid definition
    if not any(grid_kw in content_upper for grid_kw in essential_keywords['grid']):
        missing.append("grid definition")
        
    # Check for at least one property
    if not any(prop in content_upper for prop in essential_keywords['properties']):
        missing.append("properties")
        
    if missing:
        logger.error(
            f"File {file_path} appears invalid - missing: {', '.join(missing)}. "
            "At least one section and grid definition are required."
        )
        raise ValueError(f"Invalid Eclipse file - missing: {', '.join(missing)}")

    # Parse RUNSPEC section first as it may contain DIMENS
    runspec = _parse_runspec_section(content)
    
    # Parse grid properties
    grid = _parse_grid_section(content)
    
    # Parse faults if present
    faults = _parse_faults_section(content) if 'FAULTS' in content_upper else None
    
    # Parse PVT tables if present
    pvt_tables = _parse_pvt_section(content) if 'PVTO' in content or 'PVTG' in content else None
    
    # Parse regions if present
    regions = _parse_regions_section(content) if any(kw in content_upper for kw in ['REGIONS', 'EQLNUM', 'FIPNUM', 'SATNUM']) else None
    
    # Parse solution section if present
    solution = _parse_solution_section(content)
    
    # Parse summary section if present
    summary = _parse_summary_section(content)
    # Parse schedule section if present
    schedule = _parse_schedule_section(content)
    
    def _parse_regions_section(content: str) -> dict:
        """Parse Eclipse REGIONS section data.
        
        Args:
            content: Full Eclipse file content
            
        Returns:
            dict: Parsed region data including:
                - eqlnum: Equilibrium region numbers
                - fipnum: Flow simulation region numbers
                - satnum: Saturation region numbers
        """
        regions = {}
        content_upper = content.upper()
        
        def _parse_grdecl_keyword(content: str, keyword: str) -> np.ndarray:
            """
            Parse a GRDECL format keyword from Eclipse input content.
            
            Args:
                content: Full input content
                keyword: Keyword to parse (e.g. 'EQLNUM')
                
            Returns:
                numpy.ndarray: Parsed values as a 1D array
            """
            import numpy as np
            import re
            
            # Find the keyword section
            pattern = re.compile(rf'{keyword}\s*(.*?)\s*/', re.DOTALL | re.IGNORECASE)
            match = pattern.search(content)
            if not match:
                raise ValueError(f"Could not find {keyword} section in input")
                
            # Extract and clean the data
            data_str = match.group(1)
            data_str = re.sub(r'--.*?$', '', data_str, flags=re.MULTILINE)  # Remove comments
            data_str = data_str.replace('\n', ' ').replace('\t', ' ')
            
            # Convert to numpy array
            try:
                return np.array([float(x) for x in data_str.split() if x])
            except ValueError as e:
                raise ValueError(f"Failed to parse {keyword} data: {e}")
        
        # Parse EQLNUM if present
        if 'EQLNUM' in content_upper:
            regions['eqlnum'] = _parse_grdecl_keyword(content, 'EQLNUM')
            
        # Parse FIPNUM if present
        if 'FIPNUM' in content_upper:
            regions['fipnum'] = _parse_grdecl_keyword(content, 'FIPNUM')
            
        # Parse SATNUM if present
        if 'SATNUM' in content_upper:
            regions['satnum'] = _parse_grdecl_keyword(content, 'SATNUM')
            
        return regions if regions else None
    
    # Create and return ReservoirData object
    return ReservoirData(
        grid=grid,
        pvt_tables=pvt_tables,
        regions=regions,
        runspec=runspec,
        faults=faults
    )


def _parse_schedule_section(content: str) -> Dict[str, Any]:
    """Parse SCHEDULE section of Eclipse file."""
    schedule_data: Dict[str, Any] = {}
    section_match = re.search(
        r'\bSCHEDULE\b\s*\n([\s\S]*?)(?=\n\s*/\s*(?:\n|\Z|\bEND\b))',
        content,
        re.IGNORECASE | re.DOTALL
    )
    
    if not section_match:
        logger.debug("No SCHEDULE section found in Eclipse file")
        return schedule_data
        
    section_content = section_match.group(1)
    
    # Parse standard keywords
    schedule_data.update(_parse_keyword_data_blocks(section_content))
    
    # Handle well definitions
    well_matches = re.finditer(
        r'^WELSPECS\s+([^\n]+)',
        section_content,
        re.IGNORECASE | re.MULTILINE
    )
    if well_matches:
        schedule_data['WELSPECS'] = [match.group(1).strip() for match in well_matches]
    
    return schedule_data

def _parse_summary_section(content: str) -> Dict[str, Any]:
    """Parse SUMMARY section of Eclipse file."""
    summary_data: Dict[str, Any] = {}
    section_match = re.search(
        r'\bSUMMARY\b\s*\n([\s\S]*?)(?=\n\s*/\s*(?:\n|\Z|\b(?:SCHEDULE|END)\b))',
        content,
        re.IGNORECASE | re.DOTALL
    )
    
    if not section_match:
        logger.debug("No SUMMARY section found in Eclipse file")
        return summary_data
        
    section_content = section_match.group(1)
    
    # Parse standard keywords
    summary_data.update(_parse_keyword_data_blocks(section_content))
    
    # Handle special SUMMARY cases
    restart_match = re.search(r'^RESTART\s+(\S+)', section_content, re.IGNORECASE | re.MULTILINE)
    if restart_match:
        summary_data['RESTART'] = restart_match.group(1).strip()
    
    return summary_data

def _parse_solution_section(content: str) -> Dict[str, Any]:
    """Parse SOLUTION section of Eclipse file."""
    solution_data: Dict[str, Any] = {}
    section_match = re.search(
        r'\bSOLUTION\b\s*\n([\s\S]*?)(?=\n\s*/\s*(?:\n|\Z|\b(?:SCHEDULE|END)\b))',
        content,
        re.IGNORECASE | re.DOTALL
    )
    
    if not section_match:
        logger.debug("No SOLUTION section found in Eclipse file")
        return solution_data
        
    section_content = section_match.group(1)
    
    # Parse standard keywords
    solution_data.update(_parse_keyword_data_blocks(section_content))
    
    # Handle special SOLUTION cases
    equil_match = re.search(r'^EQUIL\s*\n(.*?)(?=\n\s*/\s*$)', section_content, re.IGNORECASE | re.DOTALL)
    if equil_match:
        solution_data['EQUIL'] = equil_match.group(1).strip()
    
    return solution_data

def _parse_pvt_section(content: str) -> dict:
    """Parse PVT section from Eclipse file content.
    
    Args:
        content: Raw content of Eclipse file
        
    Returns:
        Dictionary containing PVT tables data or None if no PVT sections found
    """
    pvt_tables = {}
    
    # Check for PVTO (oil) tables
    if 'PVTO' in content:
        try:
            pvto_start = content.index('PVTO')
            pvto_end = content.index('/', pvto_start)
            pvto_data = content[pvto_start:pvto_end].split('\n')[1:]  # Skip PVTO keyword
            pvt_tables['PVTO'] = [line.strip() for line in pvto_data if line.strip()]
        except ValueError as e:
            logger.warning(f"Failed to parse PVTO section: {e}")
    
    # Check for PVTG (gas) tables
    if 'PVTG' in content:
        try:
            pvtg_start = content.index('PVTG')
            pvtg_end = content.index('/', pvtg_start)
            pvtg_data = content[pvtg_start:pvtg_end].split('\n')[1:]  # Skip PVTG keyword
            pvt_tables['PVTG'] = [line.strip() for line in pvtg_data if line.strip()]
        except ValueError as e:
            logger.warning(f"Failed to parse PVTG section: {e}")
    
    return pvt_tables if pvt_tables else None
    
    if not grid and not pvt_tables and not regions:
        logger.warning(f"No valid data successfully parsed from Eclipse file: {file_path}")
        grid = grid or {} 
        pvt_tables = pvt_tables or {}
        regions = regions or {}


    return ReservoirData(
        runspec=runspec,
        grid=grid,
        pvt_tables=pvt_tables,
        regions=regions,
        solution=solution,
        summary=summary,
        schedule=schedule
    )