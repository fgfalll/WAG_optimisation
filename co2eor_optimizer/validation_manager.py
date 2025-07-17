import logging
import yaml
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

logger = logging.getLogger(__name__)

# Assumes this file is in 'co2eor_optimizer/'
PACKAGE_DIR = Path(__file__).parent.resolve()

class ValidationManager:
    """
    A singleton to load and provide validation rules for application parameters.
    Reads rules from a central YAML file.
    """
    _instance = None
    _rules = {}
    _config_path = PACKAGE_DIR / "config" / "validation_rules.yaml"

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.load_rules()
        return cls._instance

    def load_rules(self):
        """Loads the validation_rules.yaml file into memory."""
        if not self._config_path.exists():
            logger.warning(f"Validation rules file not found at '{self._config_path}'. Validation will be disabled.")
            self._rules = {}
            return

        try:
            with open(self._config_path, 'r', encoding='utf-8') as f:
                self._rules = yaml.safe_load(f)
            logger.info(f"Validation rules loaded successfully from '{self._config_path}'.")
        except (FileNotFoundError, yaml.YAMLError) as e:
            logger.error(f"Failed to load or parse validation rules: {e}", exc_info=True)
            self._rules = {}

    def get_rules_for(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Gets the validation rules for a given parameter key.
        e.g., 'ConfigWidget.EconomicParameters.interest_rate'
        """
        try:
            parts = key.split('.')
            scope = parts[0]
            param_key = '.'.join(parts[1:])
            return self._rules.get(scope, {}).get(param_key)
        except (IndexError, AttributeError):
            return None

    def validate(self, key: str, value: Any) -> Optional[Tuple[str, str]]:
        """
        Validates a value against the rules defined for its key.

        Args:
            key: The unique parameter key.
            value: The value to validate.

        Returns:
            A tuple of (level, message) if validation fails (e.g., ('warn', 'Value too high')).
            Returns None if validation passes.
        """
        rules = self.get_rules_for(key)
        if not rules:
            return None # No rules for this key

        # Type check
        rule_type = rules.get('type')
        if rule_type == 'float' or rule_type == 'int':
            try:
                numeric_value = float(value)
                if rule_type == 'int' and not numeric_value.is_integer():
                     return ('error', 'Value must be a whole number.')
            except (ValueError, TypeError):
                return ('error', 'Value must be a valid number.')
        else:
            numeric_value = None

        # Range check
        if numeric_value is not None:
            check_level = rules.get('range_check') # 'warn' or 'error'
            if not check_level:
                return None
            
            min_val = rules.get('min')
            max_val = rules.get('max')

            if min_val is not None and numeric_value < min_val:
                return (check_level, f"Value is below the suggested minimum of {min_val}.")
            if max_val is not None and numeric_value > max_val:
                return (check_level, f"Value is above the suggested maximum of {max_val}.")
        
        # All checks passed
        return None