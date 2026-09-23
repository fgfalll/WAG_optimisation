import pytest
import numpy as np
import tempfile
from pathlib import Path

from utils.las_parser import parse_las, MissingWellNameError, UNIT_CONVERSIONS
from core.data_models import WellData


@pytest.fixture
def sample_las_file():
    """Create a temporary valid LAS 2.0 file."""
    content = """~VERSION INFORMATION
 VERS.                          2.0 :   CWLS LOG ASCII STANDARD -VERSION 2.0
 WRAP.                           NO :   ONE LINE PER DEPTH STEP
~WELL INFORMATION
#MNEM.UNIT       DATA                       DESCRIPTION
#---------       ----                       -----------
 STRT.FT         1000.0000                  : START DEPTH
 STOP.FT         1005.0000                  : STOP DEPTH
 STEP.FT            1.0000                  : STEP
 NULL.           -999.2500                  : NULL VALUE
 WELL.           TEST_WELL_01               : WELL NAME
~CURVE INFORMATION
#MNEM.UNIT       API CODES                  DESCRIPTION
#---------       ---------                  -----------
 DEPT.FT                                    : 1  DEPTH
 PORO.V/V                                   : 2  POROSITY
 PERM.MD                                    : 3  PERMEABILITY
 GR.GAPI                                    : 4  GAMMA RAY
~A  DEPTH     PORO      PERM        GR
  1000.000   0.2100   150.5000   45.2000
  1001.000   0.1950   120.3000   48.1000
  1002.000   0.2200   180.0000   42.0000
  1003.000  -999.25   -999.250   50.0000
  1004.000   0.1800    95.0000   55.4000
  1005.000   0.2050   140.0000   46.8000
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".las", delete=False) as f:
        f.write(content)
        temp_path = f.name
    yield temp_path
    Path(temp_path).unlink(missing_ok=True)


@pytest.fixture
def missing_well_name_las():
    """Create a temporary LAS file without WELL header item."""
    content = """~VERSION INFORMATION
 VERS.                          2.0 :   CWLS LOG ASCII STANDARD -VERSION 2.0
 WRAP.                           NO :   ONE LINE PER DEPTH STEP
~WELL INFORMATION
 STRT.FT         1000.0000                  : START DEPTH
 STOP.FT         1001.0000                  : STOP DEPTH
 STEP.FT            1.0000                  : STEP
 NULL.           -999.2500                  : NULL VALUE
~CURVE INFORMATION
 DEPT.FT                                    : 1  DEPTH
 PORO.V/V                                   : 2  POROSITY
~A  DEPTH     PORO
  1000.000   0.2100
  1001.000   0.1950
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".las", delete=False) as f:
        f.write(content)
        temp_path = f.name
    yield temp_path
    Path(temp_path).unlink(missing_ok=True)


def test_parse_las_basic(sample_las_file):
    """Verify basic parsing extracts WellData with correct curves and well name."""
    well_data = parse_las(sample_las_file)
    assert well_data is not None
    assert isinstance(well_data, WellData)
    assert well_data.name == "TEST_WELL_01"
    assert len(well_data.depths) == 6
    assert np.isclose(well_data.depths[0], 1000.0)
    assert np.isclose(well_data.depths[-1], 1005.0)

    # Check curves
    assert "PORO" in well_data.properties
    assert "PERM" in well_data.properties
    assert "GR" in well_data.properties
    assert len(well_data.properties["PORO"]) == 6

    # Verify null value handling (replaced by NaN)
    assert np.isnan(well_data.properties["PORO"][3])
    assert np.isclose(well_data.properties["PORO"][0], 0.21)


def test_parse_las_well_name_override(sample_las_file):
    """Verify well_name_override takes precedence over file header."""
    well_data = parse_las(sample_las_file, well_name_override="CUSTOM_NAME")
    assert well_data is not None
    assert well_data.name == "CUSTOM_NAME"


def test_parse_las_missing_well_name(missing_well_name_las):
    """Verify MissingWellNameError is raised when WELL section is absent."""
    with pytest.raises(MissingWellNameError):
        parse_las(missing_well_name_las)


def test_parse_las_missing_well_name_with_override(missing_well_name_las):
    """Verify override allows parsing even when file header is missing well name."""
    well_data = parse_las(missing_well_name_las, well_name_override="OVERRIDDEN_WELL")
    assert well_data is not None
    assert well_data.name == "OVERRIDDEN_WELL"


def test_parse_las_unit_conversion(sample_las_file):
    """Verify target depth unit conversion (FT to M)."""
    well_data = parse_las(sample_las_file, depth_unit="M")
    assert well_data is not None
    assert well_data.units["DEPT"] == "M"
    # 1000 ft * 0.3048 = 304.8 m
    assert np.isclose(well_data.depths[0], 1000.0 * UNIT_CONVERSIONS["FT"]["M"])


def test_parse_las_file_not_found():
    """Verify IOError on non-existent file."""
    with pytest.raises(IOError):
        parse_las("non_existent_file_12345.las")
