import unittest
import os
import numpy as np
from pathlib import Path
from co2eor_optimizer.parsers.las_parser import parse_las, MissingWellNameError
from co2eor_optimizer.core import WellData

class TestLasParser(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_data_dir = Path(__file__).parent / 'test_data'
        cls.valid_las = cls.test_data_dir / 'valid.las'
        cls.missing_data_las = cls.test_data_dir / 'missing_data.las'
        cls.invalid_las = cls.test_data_dir / 'invalid.las'
        
        # Create test LAS files
        os.makedirs(cls.test_data_dir, exist_ok=True)
        cls._create_test_files()

    @classmethod
    def _create_test_files(cls):
        # Create valid LAS file
        with open(cls.valid_las, 'w') as f:
            f.write("""~VERSION INFORMATION
 VERS.                 2.0
~WELL
WELL.                  WellA
~CURVE INFORMATION
DEPT.M                 : 
GR  .GAPI              : Gamma Ray
RHOB.G/CC              : Bulk Density
~A
1670.0   45.3   2.65
1670.5   46.1   2.63
1671.0   47.2   2.61""")

        # Create LAS with missing data
        with open(cls.missing_data_las, 'w') as f:
            f.write("""~VERSION INFORMATION
 VERS.                 2.0
~WELL
WELL.                  WellB
~CURVE INFORMATION
DEPT.FT                : 
GR  .GAPI              : Gamma Ray
RHOB.G/CC              : Bulk Density
~A
1670.0   45.3   2.65
1670.5   -999   2.63
1671.0   47.2   -999""")

    def test_basic_parsing(self):
        """Test basic LAS file parsing"""
        result = parse_las(str(self.valid_las))
        self.assertIsInstance(result, WellData)
        self.assertEqual(result.name, "WellA")
        self.assertEqual(len(result.depths), 3)
        self.assertIn("GR", result.properties)
        self.assertIn("RHOB", result.properties)

    def test_unit_conversion(self):
        """Test depth unit conversion"""
        result = parse_las(str(self.valid_las), depth_unit="FT")
        self.assertAlmostEqual(result.depths[0], 1670.0 * 3.28084, places=4)
        self.assertEqual(result.units["DEPT"], "FT")
        self.assertAlmostEqual(result.properties["DEPT"][0], 1670.0 * 3.28084, places=4)

    def test_invalid_file(self):
        """Test error handling for invalid files"""
        with self.assertRaises(IOError):
            parse_las("nonexistent_file.las")

    def test_missing_data_handling(self):
        """Test handling of missing values"""
        result = parse_las(str(self.missing_data_las))
        # Check GR missing value (-999) was converted to NaN
        self.assertTrue(np.isnan(result.properties["GR"][1]))
        # Check RHOB missing value (-999) was converted to NaN
        self.assertTrue(np.isnan(result.properties["RHOB"][2]))

    def test_invalid_file(self):
        """Test error handling for invalid files"""
        with self.assertRaises(IOError):
            parse_las(str(self.invalid_las))

    def test_missing_well_section(self):
        """Test validation of required sections"""
        test_file = self.test_data_dir / 'empty_well.las'
        with open(test_file, 'w') as f:
            # Create LAS file with empty WELL section
            f.write("~VERSION\nVERS. 2.0\n~WELL\n~OTHER\nNOTE. Test file\n")

        # Should fail with MissingWellNameError since WELL section exists but has no name
        with self.assertRaises(MissingWellNameError) as cm:
            parse_las(str(test_file))
        with self.assertRaises(MissingWellNameError) as cm:
            parse_las(str(self.test_data_dir / 'no_well.las'))
        
        # Verify exception contains expected info
        exc = cm.exception
        self.assertEqual(exc.file_path, str(self.test_data_dir / 'no_well.las'))
        self.assertIn('version', exc.available_sections)
        self.assertIn('other', exc.available_sections)

        # Simulate user providing well name and retry
        result = parse_las(
            str(self.test_data_dir / 'no_well.las'),
            well_name_override="TEST_WELL"
        )
        self.assertEqual(result.name, "TEST_WELL")

if __name__ == '__main__':
    unittest.main()