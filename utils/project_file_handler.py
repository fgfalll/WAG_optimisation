
import json
import logging
from pathlib import Path
import numpy as np
import dataclasses
from typing import Any, Dict

from core import data_models

logger = logging.getLogger(__name__)

class ProjectEncoder(json.JSONEncoder):
    """Custom JSON encoder for project data."""
    def default(self, o: Any) -> Any:
        if dataclasses.is_dataclass(o) and not isinstance(o, type):
            d = {f.name: getattr(o, f.name) for f in dataclasses.fields(o)}
            d['_dataclass'] = o.__class__.__name__
            return d
        if isinstance(o, np.ndarray):
            return {
                '_numpy_array': True,
                'data': o.tolist(),
                'dtype': o.dtype.name
            }
        if isinstance(o, (np.integer, np.floating)):
            return o.item()
        if isinstance(o, Path):
            return str(o)
        if type(o).__name__ == 'BayesianOptimization':
            # The BayesianOptimization object from the bayes_opt library is not serializable.
            # We replace it with a placeholder.
            return "BayesianOptimization object placeholder"
        return super().default(o)

def project_decoder(data: Dict[str, Any]) -> Any:
    """Custom JSON decoder (object_hook) for project data."""
    if '_dataclass' in data:
        class_name = data.pop('_dataclass')
        cls = getattr(data_models, class_name, None)
        if cls:
            # Special backwards compatibility for nested dataclasses inside ReservoirData
            if class_name == 'ReservoirData':
                if isinstance(data.get('eos_model'), dict) and '_dataclass' not in data['eos_model']:
                    try:
                        eos_cls = getattr(data_models, 'EOSModelParameters', None)
                        if eos_cls:
                            data['eos_model'] = eos_cls(**data['eos_model'])
                    except Exception as e:
                        logger.warning(f"Failed to deserialize eos_model in ReservoirData: {e}")
                if isinstance(data.get('layer_definitions'), list):
                    layer_cls = getattr(data_models, 'LayerDefinition', None)
                    if layer_cls:
                        deserialized_layers = []
                        for l in data['layer_definitions']:
                            if isinstance(l, dict) and '_dataclass' not in l:
                                try:
                                    deserialized_layers.append(layer_cls(**l))
                                except Exception:
                                    deserialized_layers.append(l)
                            else:
                                deserialized_layers.append(l)
                        data['layer_definitions'] = deserialized_layers
                if isinstance(data.get('geostatistical_params'), dict) and '_dataclass' not in data['geostatistical_params']:
                    try:
                        geo_cls = getattr(data_models, 'GeostatisticalParams', None)
                        if geo_cls:
                            data['geostatistical_params'] = geo_cls(**data['geostatistical_params'])
                    except Exception as e:
                        logger.warning(f"Failed to deserialize geostatistical_params in ReservoirData: {e}")

            # The from_config_dict method is available on many of the dataclasses
            # and is designed to safely construct an instance from a dictionary.
            if hasattr(cls, 'from_config_dict'):
                try:
                    return cls.from_config_dict(data)
                except Exception as e:
                    logger.warning(f"Failed to deserialize {class_name} using from_config_dict: {e}. Falling back to default constructor.")
                    return cls(**data) # Fallback
            else:
                return cls(**data)
        else:
            logger.warning(f"Unknown dataclass type: {class_name}")
            return data
    elif isinstance(data, dict) and data.get('_numpy_array'):
        return np.array(data['data'], dtype=data['dtype'])
    return data

def save_project_to_tphd(data_to_save: Dict[str, Any], filepath: Path) -> bool:
    """
    Saves the project data to a .tphd file (JSON format).

    Args:
        data_to_save: A dictionary containing the project data.
        filepath: The path to save the file to.

    Returns:
        True if successful, False otherwise.
    """
    try:
        with open(filepath, 'w') as f:
            json.dump(data_to_save, f, cls=ProjectEncoder, indent=4)
        logger.info(f"Project saved successfully to {filepath}")
        return True
    except (TypeError, IOError) as e:
        logger.error(f"Failed to save project to {filepath}: {e}", exc_info=True)
        return False

def load_project_from_tphd(filepath: Path) -> Dict[str, Any]:
    """
    Loads a project from a .tphd file.

    Args:
        filepath: The path to the project file.

    Returns:
        A dictionary containing the project data, or None if loading fails.
    """
    try:
        with open(filepath, 'r') as f:
            return json.load(f, object_hook=project_decoder)
    except (IOError, json.JSONDecodeError) as e:
        logger.error(f"Failed to load project from {filepath}: {e}", exc_info=True)
        return None
