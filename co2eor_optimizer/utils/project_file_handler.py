import json
import gzip
import logging
from pathlib import Path
from typing import Dict, Any, Union, Optional
import numpy as np
import sys
from dataclasses import is_dataclass, asdict, fields

logger = logging.getLogger(__name__)

# --- Custom JSON Encoder/Decoder for NumPy arrays and Dataclasses ---
class CustomProjectEncoder(json.JSONEncoder):
    """
    Custom JSON encoder to handle:
    - NumPy arrays (converted to lists with a type marker)
    - Dataclasses (converted to dictionaries with a type marker)
    - Path objects (converted to strings)
    - Sets (converted to lists)
    """
    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return {
                "__numpy__": True,
                "array": obj.tolist(),
                "dtype": str(obj.dtype)
            }
        if is_dataclass(obj) and not isinstance(obj, type): # Handle instances, not classes
            # Store the fully qualified name for robust deserialization
            module_name = obj.__class__.__module__
            class_name = obj.__class__.__name__
            return {
                "__dataclass__": True,
                "module": module_name,
                "class": class_name,
                "data": asdict(obj) # Recursively calls default for nested complex types
            }
        if isinstance(obj, Path):
            return str(obj.as_posix()) # Store as POSIX-style string for cross-platform compatibility
        if isinstance(obj, set):
            return list(obj)
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64,
                            np.uint8, np.uint16, np.uint32, np.uint64)):
            return int(obj)
        if isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)

def project_object_hook(dct: Dict[str, Any]) -> Any:
    """
    Custom object hook for json.loads to reconstruct:
    - NumPy arrays from the custom "numpy_array_format"
    - Dataclasses from the custom "dataclass_format"
    """
    if "__numpy__" in dct and dct["__numpy__"] is True:
        try:
            return np.array(dct["array"], dtype=dct["dtype"])
        except Exception as e:
            logger.error(f"Error reconstructing numpy array: {e}. Data: {dct}")
            return dct # Return original dict if reconstruction fails
    if "__dataclass__" in dct and dct["__dataclass__"] is True:
        try:
            module_name = dct["module"]
            class_name = dct["class"]
            data = dct["data"]

            # Dynamically import the module and get the class
            # This is a potential security risk if loading untrusted files.
            # For a PhD tool, this might be acceptable if file sources are controlled.
            # A safer approach for wider distribution would be explicit registration of known dataclasses.
            module = __import__(module_name, fromlist=[class_name])
            cls = getattr(module, class_name)

            # Reconstruct nested dataclasses/numpy arrays if they were also custom encoded
            # The standard json.load with object_hook handles this recursively.
            
            # Filter data to only include fields defined in the dataclass
            # This helps with forward/backward compatibility if dataclass fields change
            field_names = {f.name for f in fields(cls)}
            filtered_data = {k: v for k, v in data.items() if k in field_names}
            
            return cls(**filtered_data)
        except (ImportError, AttributeError, TypeError) as e:
            logger.error(f"Error reconstructing dataclass {dct.get('module')}.{dct.get('class')}: {e}. Data: {dct}")
            return dct # Return original dict if reconstruction fails
    return dct

# --- Project File Handling Functions ---
def save_project_to_tphd(project_data: Dict[str, Any], filepath: Union[str, Path]) -> bool:
    """
    Saves the project data dictionary to a .tphd file (gzipped JSON).

    Args:
        project_data: A dictionary containing the entire project state.
                      This state should be serializable by the CustomProjectEncoder.
        filepath: The path (string or Path object) to save the .tphd file.

    Returns:
        True if saving was successful, False otherwise.
    """
    path = Path(filepath)
    if not path.name.endswith(".tphd"):
        path = path.with_suffix(".tphd")
        logger.info(f"Filepath adjusted to enforce .tphd extension: {path}")

    try:
        path.parent.mkdir(parents=True, exist_ok=True) # Ensure directory exists
        json_data = json.dumps(project_data, cls=CustomProjectEncoder, indent=2, ensure_ascii=False)
        with gzip.open(path, "wt", encoding="utf-8") as f: # "wt" for text mode with gzip
            f.write(json_data)
        logger.info(f"Project successfully saved to: {path.resolve()}")
        return True
    except TypeError as te:
        logger.error(f"Serialization error (type not handled by CustomProjectEncoder) saving project to {path}: {te}", exc_info=True)
    except IOError as ioe:
        logger.error(f"IOError saving project to {path}: {ioe}", exc_info=True)
    except Exception as e:
        logger.error(f"An unexpected error occurred while saving project to {path}: {e}", exc_info=True)
    return False

def load_project_from_tphd(filepath: Union[str, Path]) -> Optional[Dict[str, Any]]:
    """
    Loads project data from a .tphd file (gzipped JSON).

    Args:
        filepath: The path (string or Path object) to the .tphd file.

    Returns:
        A dictionary containing the project state, or None if loading fails.
    """
    path = Path(filepath)
    if not path.exists() or not path.is_file():
        logger.error(f"Project file not found or is not a file: {path}")
        return None
    if not path.name.endswith(".tphd"):
        logger.warning(f"File '{path}' does not have .tphd extension. Attempting to load anyway.")

    try:
        with gzip.open(path, "rt", encoding="utf-8") as f: # "rt" for text mode with gzip
            project_data = json.load(f, object_hook=project_object_hook)
        logger.info(f"Project successfully loaded from: {path.resolve()}")
        # TODO: Implement schema validation here if .tphd_schema.json is defined
        # from jsonschema import validate
        # schema = ... load schema ...
        # validate(instance=project_data, schema=schema)
        return project_data
    except gzip.BadGzipFile:
        logger.error(f"File {path} is not a valid gzip file. It might be a plain JSON or corrupted.")
        # Attempt to load as plain JSON as a fallback
        try:
            with open(path, "r", encoding="utf-8") as f_plain:
                project_data_plain = json.load(f_plain, object_hook=project_object_hook)
            logger.info(f"Successfully loaded {path} as plain JSON (fallback).")
            return project_data_plain
        except json.JSONDecodeError as jde_plain:
            logger.error(f"JSONDecodeError (plain JSON fallback) loading project from {path}: {jde_plain}", exc_info=True)
        except Exception as e_plain:
            logger.error(f"Unexpected error (plain JSON fallback) loading project from {path}: {e_plain}", exc_info=True)
    except json.JSONDecodeError as jde:
        logger.error(f"JSONDecodeError loading project from {path}: {jde}", exc_info=True)
    except IOError as ioe:
        logger.error(f"IOError loading project from {path}: {ioe}", exc_info=True)
    except Exception as e:
        logger.error(f"An unexpected error occurred while loading project from {path}: {e}", exc_info=True)
    return None

if __name__ == '__main__':
    # --- Example Usage (requires data_models.py to be importable) ---
    logging.basicConfig(level=logging.DEBUG)
    try:
        # Assuming data_models.py is in core/ relative to project_handler.py's parent
        # Adjust path if necessary for direct script execution
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from core.data_models import WellData, EconomicParameters

        # Create dummy data
        test_well = WellData(
            name="TestWell-01",
            depths=np.array([1000.0, 1010.1, 1020.2]),
            properties={
                "GR": np.array([45.0, 50.5, 48.2]),
                "PORO": np.array([0.15, 0.18, 0.16], dtype=np.float32)
            },
            units={"GR": "API", "PORO": "v/v"}
        )
        test_econ = EconomicParameters(oil_price_usd_per_bbl=88.8, discount_rate_fraction=0.12)

        sample_project_data = {
            "project_name": "EOR Test Project Alpha",
            "version": "1.0",
            "settings": {"active_tab": "DataManagement"},
            "well_data_list": [test_well],
            "economic_parameters": test_econ,
            "simulation_results": {"NPV": np.float64(12345.67), "RF": 0.25},
            "misc_paths": [Path("input/logs"), Path("output/plots")],
            "tags": {"exploration", "co2"},
            "empty_numpy": np.array([])
        }

        # Save
        save_path = Path("test_project.tphd")
        if save_project_to_tphd(sample_project_data, save_path):
            logger.info(f"Test project saved to {save_path.resolve()}")

            # Load
            loaded_data = load_project_from_tphd(save_path)
            if loaded_data:
                logger.info("Test project loaded successfully.")
                # Verify reconstruction
                loaded_well = loaded_data["well_data_list"][0]
                loaded_econ = loaded_data["economic_parameters"]

                assert isinstance(loaded_well, WellData)
                assert isinstance(loaded_well.depths, np.ndarray)
                assert np.array_equal(loaded_well.depths, test_well.depths)
                assert str(loaded_well.properties["PORO"].dtype) == 'float32'
                logger.info(f"Loaded Well Name: {loaded_well.name}")

                assert isinstance(loaded_econ, EconomicParameters)
                assert loaded_econ.oil_price_usd_per_bbl == 88.8
                logger.info(f"Loaded Econ Oil Price: {loaded_econ.oil_price_usd_per_bbl}")
                
                assert isinstance(loaded_data["simulation_results"]["NPV"], float)
                assert isinstance(loaded_data["misc_paths"][0], str) # Paths are stored as strings
                assert isinstance(loaded_data["tags"], list) # Sets are stored as lists
                assert loaded_data["empty_numpy"].size == 0


                logger.info("Data types and values verified post-load.")
            else:
                logger.error("Failed to load test project.")
        else:
            logger.error("Failed to save test project.")

    except ImportError:
        logger.warning("Could not import data_models for example usage. Skipping example.")
    except Exception as ex:
        logger.error(f"Error in example usage: {ex}", exc_info=True)