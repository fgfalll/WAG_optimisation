import logging
import numpy as np
from typing import Dict, Union, Tuple, Optional, List

logger = logging.getLogger(__name__)

class UnitsManager:
    """
    Manages unit definitions and conversions for petroleum engineering quantities.
    Supports 'SI' (International System) and 'Field' (US Oilfield) units.
    """
    # Define base units (SI is chosen as the internal base for consistency)
    _BASE_UNITS = {
        "pressure": "Pa",
        "temperature": "K", # Kelvin for internal base, though Celsius is common in SI input
        "length": "m",
        "volume": "m3",
        "density": "kg/m3",
        "viscosity": "Pa.s", # Pascal-second
        "permeability": "m2", # Darcy is often used, 1 Darcy ~ 0.9869233e-12 m²
        "rate_vol": "m3/s",
        "rate_mass": "kg/s",
        "gor": "m3/m3", # Standard m3 gas / Standard m3 oil
        "fvf": "m3/m3", # Reservoir m3 / Standard m3
        "time": "s",
        "concentration": "mol/m3", # Or fraction for compositions
        "api_gravity": "api", # API is unitless in a sense but has a formula
        "angle": "degrees"
    }

    # Conversion factors: To Base (SI) from Other Unit
    # Format: {"category": {"OtherUnit": factor_to_convert_OtherUnit_to_SI_Base}}
    _CONVERSION_FACTORS_TO_BASE = {
        "pressure": {
            "psi": 6894.76,      # 1 psi to Pa
            "bar": 100000.0,     # 1 bar to Pa
            "atm": 101325.0,     # 1 atm to Pa
            "kPa": 1000.0,       # 1 kPa to Pa
            "MPa": 1000000.0,    # 1 MPa to Pa
            "Pa": 1.0
        },
        "temperature": { # Special handling due to non-multiplicative conversion for C/F
            "°C": lambda c: c + 273.15, # Celsius to Kelvin
            "°F": lambda f: (f - 32) * 5/9 + 273.15, # Fahrenheit to Kelvin
            "K": 1.0
        },
        "length": {
            "ft": 0.3048,        # 1 foot to m
            "in": 0.0254,        # 1 inch to m
            "cm": 0.01,
            "mm": 0.001,
            "km": 1000.0,
            "m": 1.0
        },
        "volume": {
            "bbl": 0.1589873,    # 1 US oil barrel to m3
            "ft3": 0.0283168,    # 1 cubic foot to m3
            "L": 0.001,          # 1 liter to m3
            "galUS": 0.00378541, # 1 US gallon to m3
            "m3": 1.0
        },
        "density": {
            "lb/ft3": 16.0185,   # 1 lb/ft3 to kg/m3
            "g/cm3": 1000.0,     # 1 g/cm3 to kg/m3
            "kg/L": 1000.0,
            "lb/galUS": 119.826, # 1 lb/US gal to kg/m3
            "kg/m3": 1.0
        },
        "viscosity": {
            "cP": 0.001,         # 1 centipoise to Pa.s
            "Pa.s": 1.0
        },
        "permeability": {
            "D": 0.9869233e-12,  # 1 Darcy to m2
            "mD": 0.9869233e-15, # 1 milliDarcy to m2
            "m2": 1.0
        },
        "rate_vol": { # Assuming daily rates for common field units
            "bbl/d": 0.1589873 / (24 * 3600), # bbl/day to m3/s
            "ft3/d": 0.0283168 / (24 * 3600), # ft3/day to m3/s
            "MSCF/d": (1000 * 0.0283168) / (24 * 3600), # Thousand std ft3/day to m3/s
            "MMSCF/d": (1000000 * 0.0283168) / (24 * 3600), # Million std ft3/day to m3/s
            "m3/d": 1.0 / (24*3600),
            "m3/s": 1.0
        },
        "time": {
            "min": 60.0,
            "hr": 3600.0,
            "day": 24 * 3600.0,
            "year": 365.25 * 24 * 3600.0, # Average year
            "s": 1.0
        },
        "api_gravity": { # API is a formula, not a direct factor
            "api": lambda api: 141.5 / (api + 131.5) # Converts API to Specific Gravity (SG_water=1)
                                                     # To get density in kg/m3, multiply SG by water_density_kg_m3
        },
        "angle": {
            "radians": 180.0 / np.pi, # Radians to Degrees
            "degrees": 1.0
        }
        # GOR, FVF are ratios, often unitless or expressed as vol/vol. Base SI is m3/m3.
        # Field GOR: scf/STB. Factor: (0.0283168 m3/scf) / (0.1589873 m3/STB) = 0.1781076 m3/m3 per scf/STB
        # Composition fractions (mol/mol, wt/wt) are unitless.
    }
    _WATER_DENSITY_KG_M3 = 997.0 # Approximate, for API to density conversion

    # Conversion factors: To Other Unit from Base (SI)
    # Inverse of _CONVERSION_FACTORS_TO_BASE (handled dynamically)
    _CONVERSION_FACTORS_FROM_BASE = {} # Populated in __init__

    # Standard display units for different systems
    _SYSTEM_DISPLAY_UNITS = {
        "SI": {
            "pressure": "kPa", "temperature": "°C", "length": "m", "volume": "m3",
            "density": "kg/m3", "viscosity": "cP", "permeability": "mD",
            "rate_vol": "m3/d", "gor": "m3/m3", "fvf": "m3/m3", "time": "day",
            "api_gravity": "api", "angle": "degrees"
        },
        "Field": {
            "pressure": "psi", "temperature": "°F", "length": "ft", "volume": "bbl",
            "density": "lb/ft3", "viscosity": "cP", "permeability": "mD",
            "rate_vol": "bbl/d", # For oil
            "rate_gas_vol": "MSCF/d", # For gas
            "gor": "scf/STB", "fvf": "bbl/STB", "time": "day",
            "api_gravity": "api", "angle": "degrees"
        }
    }

    def __init__(self):
        self._populate_from_base_conversions()

    def _populate_from_base_conversions(self):
        """Populates _CONVERSION_FACTORS_FROM_BASE using _CONVERSION_FACTORS_TO_BASE."""
        for category, cat_factors in self._CONVERSION_FACTORS_TO_BASE.items():
            self._CONVERSION_FACTORS_FROM_BASE[category] = {}
            base_unit_for_cat = self._BASE_UNITS[category]
            for unit, factor_or_func in cat_factors.items():
                if unit == base_unit_for_cat: # e.g. Pa -> Pa
                    self._CONVERSION_FACTORS_FROM_BASE[category][unit] = 1.0
                    continue

                if callable(factor_or_func): # Special handling for temperature, API
                    if category == "temperature":
                        if unit == "°C": # K to C
                            self._CONVERSION_FACTORS_FROM_BASE[category][unit] = lambda k: k - 273.15
                        elif unit == "°F": # K to F
                            self._CONVERSION_FACTORS_FROM_BASE[category][unit] = lambda k: (k - 273.15) * 9/5 + 32
                    elif category == "api_gravity":
                        if unit == "api": # SG to API
                             self._CONVERSION_FACTORS_FROM_BASE[category][unit] = lambda sg: (141.5 / sg) - 131.5
                else: # Multiplicative factor
                    if abs(factor_or_func) > 1e-12: # Avoid division by zero
                         self._CONVERSION_FACTORS_FROM_BASE[category][unit] = 1.0 / factor_or_func
                    else:
                         self._CONVERSION_FACTORS_FROM_BASE[category][unit] = 0 # Or raise error

    def _get_category_for_unit(self, unit: str) -> Optional[str]:
        """Finds the category a unit belongs to."""
        unit_lower = unit.lower()
        for category, units_map in self._CONVERSION_FACTORS_TO_BASE.items():
            if unit_lower in (u.lower() for u in units_map.keys()):
                return category
        # Check base units as well
        for category, base_unit_val in self._BASE_UNITS.items():
            if unit_lower == base_unit_val.lower():
                return category
        return None

    def convert(self,
                value: Union[float, np.ndarray],
                from_unit: str,
                to_unit: str,
                category: Optional[str] = None) -> Union[float, np.ndarray]:
        """
        Converts a value from one unit to another.

        Args:
            value: The numerical value or NumPy array to convert.
            from_unit: The unit of the input value (e.g., "psi", "ft").
            to_unit: The target unit to convert to.
            category: Optional. The physical quantity category (e.g., "pressure", "length").
                      If None, the manager will try to infer it.

        Returns:
            The converted value or NumPy array.

        Raises:
            ValueError: If units are incompatible or not found.
        """
        if from_unit == to_unit:
            return value

        cat_from = category or self._get_category_for_unit(from_unit)
        cat_to = category or self._get_category_for_unit(to_unit)

        if not cat_from:
            raise ValueError(f"Unit '{from_unit}' not recognized or category not inferable.")
        if not cat_to:
            raise ValueError(f"Unit '{to_unit}' not recognized or category not inferable.")
        if cat_from != cat_to:
            raise ValueError(f"Cannot convert between different categories: '{cat_from}' ({from_unit}) to '{cat_to}' ({to_unit}).")

        # Step 1: Convert from_unit to base SI unit for the category
        base_unit_si = self._BASE_UNITS[cat_from]
        value_in_base_si: Union[float, np.ndarray]

        if from_unit == base_unit_si:
            value_in_base_si = value
        else:
            factors_to_base = self._CONVERSION_FACTORS_TO_BASE.get(cat_from, {})
            conversion_factor_or_func = factors_to_base.get(from_unit)
            if conversion_factor_or_func is None:
                raise ValueError(f"No conversion factor defined for '{from_unit}' in category '{cat_from}'. Known: {list(factors_to_base.keys())}")

            if callable(conversion_factor_or_func):
                value_in_base_si = conversion_factor_or_func(value)
            else:
                value_in_base_si = value * conversion_factor_or_func
        
        # Handle special case: API Gravity to density
        if cat_from == "api_gravity" and from_unit == "api" and base_unit_si == "api": # Base for API is "api" meaning SG
             # value_in_base_si is now SG. If target is density, convert SG to density
            if self._get_category_for_unit(to_unit) == "density":
                if to_unit == "kg/m3": return value_in_base_si * self._WATER_DENSITY_KG_M3
                # Add other density units if needed, converting from SG * water_density first

        # Step 2: Convert from base SI unit to to_unit
        if to_unit == base_unit_si:
            return value_in_base_si
        else:
            factors_from_base = self._CONVERSION_FACTORS_FROM_BASE.get(cat_from, {})
            conversion_factor_or_func_to_target = factors_from_base.get(to_unit)
            if conversion_factor_or_func_to_target is None:
                raise ValueError(f"No conversion factor defined from base for '{to_unit}' in category '{cat_from}'. Known: {list(factors_from_base.keys())}")

            if callable(conversion_factor_or_func_to_target):
                return conversion_factor_or_func_to_target(value_in_base_si)
            else:
                return value_in_base_si * conversion_factor_or_func_to_target


    def get_display_unit(self, category: str, system: str = "Field") -> str:
        """
        Gets the standard display unit for a given category and unit system.

        Args:
            category: The physical quantity category (e.g., "pressure").
            system: The unit system ("SI" or "Field"). Defaults to "Field".

        Returns:
            The display unit string.

        Raises:
            ValueError: If the category or system is unknown.
        """
        system_units = self._SYSTEM_DISPLAY_UNITS.get(system)
        if not system_units:
            raise ValueError(f"Unknown unit system: '{system}'. Supported: {list(self._SYSTEM_DISPLAY_UNITS.keys())}")
        
        display_unit = system_units.get(category)
        if not display_unit:
            # Fallback to base SI unit if category not in specific system map but is known
            base_unit = self._BASE_UNITS.get(category)
            if base_unit:
                logger.warning(f"Category '{category}' not explicitly in '{system}' display map. Using base SI unit: '{base_unit}'.")
                return base_unit
            raise ValueError(f"Unknown category '{category}' for display units. Known: {list(self._BASE_UNITS.keys())}")
        return display_unit

    def get_available_units_for_category(self, category: str) -> List[str]:
        """Returns a list of all known units for a given category."""
        if category not in self._CONVERSION_FACTORS_TO_BASE:
            if category not in self._BASE_UNITS:
                raise ValueError(f"Unknown category: {category}")
            return [self._BASE_UNITS[category]] # Only base unit is known
        
        units = set(self._CONVERSION_FACTORS_TO_BASE[category].keys())
        units.add(self._BASE_UNITS[category])
        return sorted(list(units))

    def get_all_categories(self) -> List[str]:
        """Returns a list of all known categories."""
        return sorted(list(self._BASE_UNITS.keys()))

# Global instance for convenience
units_manager = UnitsManager()

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)

    um = UnitsManager()

    # Test basic conversions
    psi_val = 14.7
    pa_val = um.convert(psi_val, "psi", "Pa", "pressure")
    kpa_val = um.convert(psi_val, "psi", "kPa", "pressure")
    logger.info(f"{psi_val} psi is {pa_val:.2f} Pa or {kpa_val:.2f} kPa")
    assert abs(kpa_val - 101.325) < 0.1

    temp_f = 68.0
    temp_c = um.convert(temp_f, "°F", "°C", "temperature")
    temp_k_from_f = um.convert(temp_f, "°F", "K", "temperature")
    temp_k_from_c = um.convert(temp_c, "°C", "K", "temperature")
    logger.info(f"{temp_f}°F is {temp_c:.1f}°C. From F to K: {temp_k_from_f:.2f}K. From C to K: {temp_k_from_c:.2f}K")
    assert abs(temp_c - 20.0) < 0.01
    assert abs(temp_k_from_f - temp_k_from_c) < 0.01

    # Test inverse temperature
    temp_c_inv = um.convert(temp_k_from_f, "K", "°C", "temperature")
    temp_f_inv = um.convert(temp_k_from_f, "K", "°F", "temperature")
    logger.info(f"{temp_k_from_f:.2f}K is {temp_c_inv:.1f}°C or {temp_f_inv:.1f}°F")
    assert abs(temp_c_inv - temp_c) < 0.01
    assert abs(temp_f_inv - temp_f) < 0.01
    
    # Test length
    feet = 100.0
    meters = um.convert(feet, "ft", "m", "length")
    logger.info(f"{feet} ft is {meters:.2f} m")
    assert abs(meters - 30.48) < 0.001

    # Test permeability
    md_val = 100.0
    d_val_from_md = um.convert(md_val, "mD", "D", "permeability")
    m2_val = um.convert(md_val, "mD", "m2", "permeability")
    logger.info(f"{md_val} mD is {d_val_from_md} D, or {m2_val:.3e} m2")
    assert abs(d_val_from_md - 0.1) < 1e-9

    # Test API Gravity to SG (which is the base for 'api_gravity' category)
    api_g = 30.0
    sg = um.convert(api_g, "api", um._BASE_UNITS["api_gravity"], "api_gravity")
    expected_sg = 141.5 / (30 + 131.5)
    logger.info(f"{api_g} API is {sg:.4f} SG (Specific Gravity)")
    assert abs(sg - expected_sg) < 1e-5

    # Test SG back to API
    api_g_inv = um.convert(sg, um._BASE_UNITS["api_gravity"], "api", "api_gravity")
    logger.info(f"{sg:.4f} SG is {api_g_inv:.1f} API")
    assert abs(api_g_inv - api_g) < 0.01

    # Test display units
    logger.info(f"Field display unit for pressure: {um.get_display_unit('pressure', 'Field')}")
    logger.info(f"SI display unit for temperature: {um.get_display_unit('temperature', 'SI')}")
    assert um.get_display_unit('pressure', 'Field') == "psi"
    assert um.get_display_unit('temperature', 'SI') == "°C"

    # Test available units
    logger.info(f"Available units for 'viscosity': {um.get_available_units_for_category('viscosity')}")
    logger.info(f"All categories: {um.get_all_categories()}")

    # Test NumPy array conversion
    pressures_psi = np.array([1000.0, 2000.0, 3000.0])
    pressures_kpa = um.convert(pressures_psi, "psi", "kPa", "pressure")
    logger.info(f"Pressures {pressures_psi} psi convert to {pressures_kpa} kPa")
    expected_kpa = pressures_psi * 6.89476
    assert np.allclose(pressures_kpa, expected_kpa)

    logger.info("UnitsManager tests passed.")