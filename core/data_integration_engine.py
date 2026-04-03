"""
Data Integration Engine for CO2 EOR Simulator
Integrates data management widget output with physics engines,
ensuring complete, valid datasets without hardcoding.
"""

import logging
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import asdict
import json
from pathlib import Path

from core.data_models import (
    ReservoirData,
    PVTProperties,
    EORParameters,
    EconomicParameters,
    OperationalParameters,
    WellData,
    LayerDefinition,
    GeostatisticalParams,
    EOSModelParameters,
    CCUSParameters,
    HuffNPuffParams,
    SWAGParams,
    TaperedInjectionParams,
    PulsedInjectionParams,
    EmpiricalFittingParameters,
    PhysicalConstants,
)
from core.engine_factory import EngineFactory
from core.geology.geostatistical_modeling import (
    create_geostatistical_grid,
    create_facies_based_grid,
)

_PHYS_CONSTANTS = PhysicalConstants()

logger = logging.getLogger(__name__)


class DataIntegrationEngine:
    """
    Central engine for integrating data management widget output
    with physics engines, ensuring complete and validated datasets.
    """

    def __init__(self):
        self.engine_factory = EngineFactory()
        self.data_validator = DataValidator()
        self.unit_converter = UnitConverter()
        self.preprocessor = DataPreprocessor()

        # Cache for processed data
        self._data_cache = {}

    def process_and_validate_dataset(self, widget_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process and validate complete dataset from data management widget
        """
        try:
            # Preprocess data
            preprocessed_data = self.preprocessor.preprocess_all(widget_data)

            # Initialize validation results
            validation_results = {
                "is_valid": True,
                "errors": [],
                "component_results": {},
                "processed_data": preprocessed_data,
            }

            # Validate reservoir parameters
            if "reservoir_parameters" in preprocessed_data:
                res_valid, res_errors = self._validate_reservoir_parameters(
                    preprocessed_data["reservoir_parameters"]
                )
                validation_results["component_results"]["reservoir"] = {
                    "valid": res_valid,
                    "errors": res_errors,
                }
                if not res_valid:
                    validation_results["is_valid"] = False
                    validation_results["errors"].extend(res_errors)

            # Validate PVT parameters
            if "pvt_parameters" in preprocessed_data:
                pvt_valid, pvt_errors = self._validate_pvt_parameters(
                    preprocessed_data["pvt_parameters"]
                )
                validation_results["component_results"]["pvt"] = {
                    "valid": pvt_valid,
                    "errors": pvt_errors,
                }
                if not pvt_valid:
                    validation_results["is_valid"] = False
                    validation_results["errors"].extend(pvt_errors)

            # Validate geostatistical parameters
            if preprocessed_data.get("geostatistical_params"):
                geo_valid, geo_errors = self._validate_geostatistical_parameters(
                    preprocessed_data["geostatistical_params"]
                )
                validation_results["component_results"]["geostatistical"] = {
                    "valid": geo_valid,
                    "errors": geo_errors,
                }
                if not geo_valid:
                    validation_results["is_valid"] = False
                    validation_results["errors"].extend(geo_errors)

            # Validate well data
            if "well_data" in preprocessed_data:
                well_valid, well_errors = self._validate_well_data(preprocessed_data["well_data"])
                validation_results["component_results"]["wells"] = {
                    "valid": well_valid,
                    "errors": well_errors,
                }
                if not well_valid:
                    validation_results["is_valid"] = False
                    validation_results["errors"].extend(well_errors)

            # Validate geomechanics parameters if enabled
            if preprocessed_data.get("geomechanics_enabled", False):
                geomech_params = preprocessed_data.get("geomechanics_parameters", {})
                if geomech_params:
                    geomech_valid, geomech_errors = self._validate_geomechanics_parameters(
                        geomech_params
                    )
                    validation_results["component_results"]["geomechanics"] = {
                        "valid": geomech_valid,
                        "errors": geomech_errors,
                    }
                    if not geomech_valid:
                        validation_results["is_valid"] = False
                        validation_results["errors"].extend(geomech_errors)

            # Apply geostatistical modeling if validation passes
            if validation_results["is_valid"]:
                processed_data = self._apply_geostatistical_modeling(preprocessed_data)
                
                # Create GeomechanicsParameters object if enabled
                geomech_obj = self._create_geomechanics_parameters(preprocessed_data)
                if geomech_obj:
                    processed_data["geomechanics_parameters_obj"] = geomech_obj
                
                validation_results["processed_data"] = processed_data

            return validation_results

        except Exception as e:
            # Import global error handler
            from error_handler import report_caught_error, ErrorSeverity, ErrorCategory

            # Report the error properly instead of just logging
            report_caught_error(
                operation="process and validate dataset",
                exception=e,
                context={
                    "dataset_type": type(widget_data).__name__,
                    "dataset_keys": list(widget_data.keys())
                    if isinstance(widget_data, dict)
                    else "non-dict",
                    "validation_stage": "comprehensive",
                    "component_results_expected": True,
                },
                user_action_suggested="Check dataset format and ensure all required fields are present. Verify that reservoir parameters, PVT data, and well data are properly formatted.",
                show_dialog=True,
                severity=ErrorSeverity.ERROR,
                category=ErrorCategory.DATA,
            )

            logger.error(f"Error in process_and_validate_dataset: {e}")
            return {
                "is_valid": False,
                "errors": [f"Processing error: {str(e)}"],
                "component_results": {},
                "processed_data": None,
            }

    def _apply_geostatistical_modeling(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply geostatistical modeling to reservoir data"""
        try:
            # Check if geostatistical modeling is enabled and params are available
            if not data.get("geostatistical_enabled", False):
                logger.info("Geostatistical modeling not enabled, skipping")
                data["generated_properties"] = None
                data["heterogeneous_reservoir"] = False
                return data

            if data.get("geostatistical_params") is None:
                logger.info("No geostatistical parameters provided, skipping")
                data["generated_properties"] = None
                data["heterogeneous_reservoir"] = False
                return data

            logger.info("Applying geostatistical modeling")

            geo_params = GeostatisticalParams(**data["geostatistical_params"])
            res_params = data["reservoir_parameters"]

            # Get grid dimensions
            nx, ny, nz = (
                res_params["grid_dimensions"]["nx"],
                res_params["grid_dimensions"]["ny"],
                res_params["grid_dimensions"]["nz"],
            )

            # Generate heterogeneous properties
            geostatistical_grid = self._generate_heterogeneous_properties((nx, ny, nz), geo_params)

            # Update data with generated properties
            data["generated_properties"] = geostatistical_grid
            data["heterogeneous_reservoir"] = True

            logger.info(f"Generated heterogeneous properties for {nx}x{ny}x{nz} grid")

        except Exception as e:
            logger.error(f"Error applying geostatistical modeling: {e}")
            data["generated_properties"] = None
            data["heterogeneous_reservoir"] = False

        return data

    def create_engine_data_structures(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate complete data structures for physics engines.

        Public method for use by DataManagementWidget to create properly
        structured engine data models compatible with surrogate engine.

        Args:
            data: Dictionary containing reservoir_parameters, pvt_parameters, eor_parameters,
                   operational_parameters, economic_parameters, and well_data

        Returns:
            Dictionary with ReservoirData, PVTProperties, EORParameters,
            OperationalParameters, EconomicParameters, and WellData
        """
        engine_data = {}

        # Generate ReservoirData
        engine_data["reservoir_data"] = self._create_reservoir_data(data)

        # Generate PVTProperties
        engine_data["pvt_properties"] = self._create_pvt_properties(data)

        # Generate EORParameters
        engine_data["eor_parameters"] = self._create_eor_parameters(data)

        # Generate OperationalParameters
        engine_data["operational_parameters"] = self._create_operational_parameters(data)

        # Generate EconomicParameters
        engine_data["economic_parameters"] = self._create_economic_parameters(data)

        # Process well data
        engine_data["well_data"] = self._process_well_data(data.get("well_data", []))

        # Generate fitting parameters for surrogate engine
        engine_data["fitting_parameters"] = self._create_fitting_parameters(data)

        return engine_data

    def _create_fitting_parameters(self, data: Dict[str, Any]) -> EmpiricalFittingParameters:
        """Create EmpiricalFittingParameters for surrogate engine tuning"""
        fitting_data = data.get("fitting_parameters", {})

        return EmpiricalFittingParameters(
            # Fluid composition
            c7_plus_fraction=fitting_data.get("c7_plus_fraction", 0.35),

            # Miscibility transition parameters
            alpha_base=fitting_data.get("alpha_base", 1.0),
            miscibility_window=fitting_data.get("miscibility_window", 0.011),

            # Production dynamics
            breakthrough_time_years=fitting_data.get("breakthrough_time_years", 1.5),
            trapping_efficiency=fitting_data.get("trapping_efficiency", 0.4),

            # Initial conditions
            initial_gor_scf_per_stb=fitting_data.get("initial_gor_scf_per_stb", 500.0),

            # Mobility and mixing
            transverse_mixing_calibration=fitting_data.get("transverse_mixing_calibration", 0.5),
            omega_tl=fitting_data.get("omega_tl", 0.6),

            # Relative permeability endpoints (Corey parameters)
            k_ro_0=fitting_data.get("k_ro_0", 0.8),
            k_rg_0=fitting_data.get("k_rg_0", 0.3),
            n_o=fitting_data.get("n_o", 2.0),
            n_g=fitting_data.get("n_g", 2.0),
        )

    def _generate_engine_data_structures(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate complete data structures for physics engines"""
        engine_data = {}

        # Generate ReservoirData
        engine_data["reservoir_data"] = self._create_reservoir_data(data)

        # Generate PVTProperties
        engine_data["pvt_properties"] = self._create_pvt_properties(data)

        # Generate EORParameters
        engine_data["eor_parameters"] = self._create_eor_parameters(data)

        # Generate OperationalParameters
        engine_data["operational_parameters"] = self._create_operational_parameters(data)

        # Generate EconomicParameters
        engine_data["economic_parameters"] = self._create_economic_parameters(data)

        # Process well data
        engine_data["well_data"] = self._process_well_data(data["well_data"])

        return engine_data

    def _create_reservoir_data(self, data: Dict[str, Any]) -> ReservoirData:
        """Create complete ReservoirData object"""
        res_params = data["reservoir_parameters"]
        pvt_params = data["pvt_parameters"]

        # Grid parameters
        nx, ny, nz = (
            res_params["grid_dimensions"]["nx"],
            res_params["grid_dimensions"]["ny"],
            res_params["grid_dimensions"]["nz"],
        )
        dx, dy, dz = (
            res_params["block_sizes"]["dx"],
            res_params["block_sizes"]["dy"],
            res_params["block_sizes"]["dz"],
        )

        n_cells = nx * ny * nz

        # Create coordinate arrays
        x_coords = np.repeat(np.arange(nx), ny * nz) * dx
        y_coords = np.tile(np.repeat(np.arange(ny), nz), nx) * dy
        z_coords = np.tile(np.arange(nz), nx * ny) * dz

        # Use generated properties if available, otherwise use uniform
        if data.get("generated_properties"):
            props = data["generated_properties"]
            porosity = props["porosity"]
            perm_x = props["permeability_x"]
            perm_y = props["permeability_y"]
            perm_z = props["permeability_z"]
        else:
            porosity = np.full((nx, ny, nz), 0.2)
            perm_x = np.full((nx, ny, nz), 100.0)
            perm_y = np.full((nx, ny, nz), 100.0)
            perm_z = np.full((nx, ny, nz), 10.0)

        # Create grid dictionary
        grid = {
            "COORD-X": x_coords,
            "COORD-Y": y_coords,
            "COORD-Z": z_coords,
            "DX": np.full(n_cells, dx),
            "DY": np.full(n_cells, dy),
            "DZ": np.full(n_cells, dz),
            "PORO": porosity.flatten(),
            "PERMX": perm_x.flatten(),
            "PERMY": perm_y.flatten(),
            "PERMZ": perm_z.flatten(),
        }

        # Create PVT tables
        pressure_points = np.linspace(1000, 6000, 50)
        pvt_tables = {
            "PRESSURE": pressure_points,
            "OIL_FVF": 1.2 + 0.0001 * (pressure_points - 4000),
            "OIL_VISC": pvt_params.get("oil_viscosity_cp", 2.0)
            * np.exp(-0.0003 * (pressure_points - 4000)),
            "GAS_FVF": 0.005 * (4000 / pressure_points),
            "CO2_VISC": pvt_params.get("gas_viscosity_cp", 0.02)
            * np.exp(-0.0002 * (pressure_points - 4000)),
        }

        return ReservoirData(
            grid=grid,
            pvt_tables=pvt_tables,
            ooip_stb=res_params["ooip_stb"],
            initial_pressure=res_params["initial_pressure"],
            temperature=res_params["temperature"],
            rock_compressibility=res_params["rock_compressibility"],
            average_porosity=np.mean(porosity),
            average_permeability=np.mean(perm_x),
            initial_water_saturation=res_params["initial_water_saturation"],
            thickness_ft=nz * dz / 0.3048,
            area_acres=(nx * dx * ny * dy) / 4046.86,
            length_ft=(nx * dx) / 0.3048,
        )

    def _create_pvt_properties(self, data: Dict[str, Any]) -> PVTProperties:
        """Create complete PVTProperties object"""
        pvt_params = data["pvt_parameters"]

        # Generate pressure points
        pressure_points = np.linspace(1000, 6000, 50)

        # Generate PVT correlations based on parameters
        api = pvt_params.get("api_gravity", 35.0)
        gas_sg = pvt_params.get("gas_specific_gravity", 0.65)
        oil_visc = pvt_params.get("oil_viscosity_cp", 2.0)
        gas_visc = pvt_params.get("gas_viscosity_cp", 0.02)
        water_visc = pvt_params.get("water_viscosity_cp", 0.5)

        # Generate oil FVF correlation
        oil_fvf = 1.2 + 0.0001 * (pressure_points - 4000) + 0.00005 * (api - 35)

        # Generate oil viscosity correlation
        oil_viscosity = oil_visc * np.exp(-0.0003 * (pressure_points - 4000))

        # Generate gas FVF
        gas_fvf = 0.005 * (4000 / pressure_points) * (gas_sg / 0.65)

        # Generate CO2 viscosity
        co2_viscosity = gas_visc * np.exp(-0.0002 * (pressure_points - 4000))

        # Generate solution GOR
        rs = 200 * np.minimum(1.0, pressure_points / 4000) * (1 + 0.1 * (gas_sg - 0.65))

        return PVTProperties(
            pressure_points=pressure_points,
            oil_fvf=oil_fvf,
            oil_viscosity=oil_viscosity,
            gas_fvf=gas_fvf,
            co2_viscosity=co2_viscosity,
            rs=rs,
            pvt_type="black_oil",
            gas_specific_gravity=gas_sg,
            temperature=data["reservoir_parameters"]["temperature"],
            api_gravity=api,
            oil_viscosity_cp=oil_visc,
            gas_viscosity_cp=gas_visc,
            water_viscosity_cp=water_visc,
        )

    def _create_eor_parameters(self, data: Dict[str, Any]) -> EORParameters:
        """Create complete EORParameters object"""
        eor_data = data.get("eor_parameters", {})

        # Create injection scheme parameters
        injection_scheme = eor_data.get("injection_scheme", "continuous")
        swag_params = None
        huff_n_puff_params = None
        tapered_params = None
        pulsed_params = None

        if injection_scheme == "swag":
            swag_params = SWAGParams(
                water_gas_ratio=eor_data.get("swag_water_gas_ratio", 1.0),
                simultaneous_injection=eor_data.get("swag_simultaneous_injection", True),
                mixing_efficiency=eor_data.get("swag_mixing_efficiency", 0.8),
            )
        elif injection_scheme == "huff_n_puff":
            huff_n_puff_params = HuffNPuffParams(
                cycle_length_days=eor_data.get("huff_n_puff_cycle_length_days", 30),
                injection_period_days=eor_data.get("huff_n_puff_injection_period_days", 10),
                soaking_period_days=eor_data.get("huff_n_puff_soaking_period_days", 5),
                production_period_days=eor_data.get("huff_n_puff_production_period_days", 15),
                max_cycles=eor_data.get("huff_n_puff_max_cycles", 10),
            )

        return EORParameters(
            injection_scheme=injection_scheme,
            injection_rate=eor_data.get("injection_rate", 5000.0),
            swag=swag_params,
            huff_n_puff=huff_n_puff_params,
            tapered=tapered_params,
            pulsed=pulsed_params,
            mobility_ratio=eor_data.get("mobility_ratio", 5.0),
            target_pressure_psi=eor_data.get("target_pressure_psi", 3000.0),
            max_pressure_psi=eor_data.get("max_pressure_psi", 6000.0),
            min_injection_rate_mscfd=eor_data.get("min_injection_rate_mscfd", 5000.0),
            max_injection_rate_mscfd=eor_data.get("max_injection_rate_mscfd", 100000.0),
            sor=eor_data.get("sor", 0.25),
            co2_density_tonne_per_mscf=eor_data.get("co2_density_tonne_per_mscf", 0.053),
            s_gc=eor_data.get("s_gc", 0.05),
            n_o=eor_data.get("n_o", 2.0),
            n_g=eor_data.get("n_g", 2.0),
            s_wc=eor_data.get("s_wc", 0.2),
            s_orw=eor_data.get("s_orw", 0.2),
            n_w=eor_data.get("n_w", 2.0),
            n_ow=eor_data.get("n_ow", 2.0),
            productivity_index=eor_data.get("productivity_index", 5.0),
            wellbore_pressure=eor_data.get("wellbore_pressure", 500.0),
            well_shut_in_threshold_bpd=eor_data.get("well_shut_in_threshold_bpd", 10.0),
            max_injector_bhp_psi=eor_data.get("max_injector_bhp_psi", 8000.0),
            timestep_days=eor_data.get("timestep_days", 30.44),
        )

    def _create_operational_parameters(self, data: Dict[str, Any]) -> OperationalParameters:
        """Create complete OperationalParameters object"""
        op_data = data.get("operational_parameters", {})

        return OperationalParameters(
            project_lifetime_years=op_data.get("project_lifetime_years", 15),
            time_resolution=op_data.get("time_resolution", "yearly"),
            target_objective_name=op_data.get("target_objective_name", None),
            target_objective_value=op_data.get("target_objective_value", None),
            recovery_model_selection=op_data.get("recovery_model_selection", "hybrid"),
            target_tolerance=op_data.get("target_tolerance", 0.05),
        )

    def _create_economic_parameters(self, data: Dict[str, Any]) -> EconomicParameters:
        """Create complete EconomicParameters object"""
        econ_data = data.get("economic_parameters", {})

        return EconomicParameters(
            oil_price_usd_per_bbl=econ_data.get("oil_price_usd_per_bbl", 70.0),
            co2_purchase_cost_usd_per_tonne=econ_data.get("co2_cost_per_ton", 50.0),
            co2_recycle_cost_usd_per_tonne=econ_data.get("co2_recycle_cost", 15.0),
            co2_storage_credit_usd_per_tonne=econ_data.get("co2_storage_credit", 25.0),
            water_injection_cost_usd_per_bbl=econ_data.get("water_injection_cost_usd_per_bbl", 1.0),
            water_disposal_cost_usd_per_bbl=econ_data.get("water_disposal_cost_usd_per_bbl", 2.0),
            discount_rate_fraction=econ_data.get("discount_rate_fraction", 0.10),
            capex_usd=econ_data.get("capex_usd", 5_000_000.0),
            fixed_opex_usd_per_year=econ_data.get("fixed_opex_usd_per_year", 200_000.0),
            variable_opex_usd_per_bbl=econ_data.get("variable_opex_usd_per_bbl", 5.0),
            carbon_tax_usd_per_tonne=econ_data.get("carbon_tax_usd_per_tonne", 0.0),
        )

    def _process_well_data(self, well_data_list: List[Dict[str, Any]]) -> List[WellData]:
        """Process well data into WellData objects"""
        processed_wells = []

        for well_info in well_data_list:
            well_data = WellData(
                name=well_info.get("name", f"Well_{len(processed_wells) + 1}"),
                depths=np.array([well_info.get("z", 0.0)]),
                properties={},
                units={},
                metadata={
                    "type": well_info.get("type", "producer"),
                    "status": well_info.get("status", "active"),
                    "x": well_info.get("x", 0.0),
                    "y": well_info.get("y", 0.0),
                    "z": well_info.get("z", 0.0),
                },
                well_path=np.array(
                    [
                        [
                            well_info.get("x", 0.0),
                            well_info.get("y", 0.0),
                            well_info.get("z", 0.0),
                        ]
                    ]
                ),
            )
            processed_wells.append(well_data)

        return processed_wells

    def _validate_engine_compatibility(self, engine_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate data compatibility with both engines"""
        compatibility = {
            "simple_engine": {"compatible": True, "issues": []},
            "detailed_engine": {"compatible": True, "issues": []},
        }

        try:
            # Test simple engine compatibility
            simple_engine = self.engine_factory.create_engine("simple")
            simple_validation = simple_engine.validate_parameters(
                engine_data["reservoir_data"], engine_data["eor_parameters"]
            )

            if not all(simple_validation.values()):
                compatibility["simple_engine"]["compatible"] = False
                compatibility["simple_engine"]["issues"] = [
                    f"Parameter {param}: {status}"
                    for param, status in simple_validation.items()
                    if not status
                ]

            # Test detailed engine compatibility
            detailed_engine = self.engine_factory.create_engine("detailed")
            detailed_validation = detailed_engine.validate_parameters(
                engine_data["reservoir_data"], engine_data["eor_parameters"]
            )

            if not all(detailed_validation.values()):
                compatibility["detailed_engine"]["compatible"] = False
                compatibility["detailed_engine"]["issues"] = [
                    f"Parameter {param}: {status}"
                    for param, status in detailed_validation.items()
                    if not status
                ]

        except Exception as e:
            logger.error(f"Error testing engine compatibility: {e}")
            compatibility["simple_engine"]["compatible"] = False
            compatibility["detailed_engine"]["compatible"] = False
            compatibility["simple_engine"]["issues"].append(f"Engine test failed: {e}")
            compatibility["detailed_engine"]["issues"].append(f"Engine test failed: {e}")

        return compatibility

    # Helper validation methods
    def _validate_reservoir_parameters(self, params: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate reservoir parameters"""
        errors = []

        # Check grid dimensions
        grid_dims = params.get("grid_dimensions", {})
        if not all(
            isinstance(grid_dims.get(dim, 0), int) and grid_dims.get(dim, 0) > 0
            for dim in ["nx", "ny", "nz"]
        ):
            errors.append("Grid dimensions must be positive integers")

        # Check block sizes
        block_sizes = params.get("block_sizes", {})
        if not all(block_sizes.get(size, 0) > 0 for size in ["dx", "dy", "dz"]):
            errors.append("Block sizes must be positive numbers")

        # Check pressure and temperature ranges
        if not (500 <= params.get("initial_pressure", 0) <= 20000):
            errors.append("Initial pressure must be between 500 and 20000 psi")

        if not (50 <= params.get("temperature", 0) <= 400):
            errors.append("Temperature must be between 50 and 400°F")

        return len(errors) == 0, errors

    def _validate_pvt_parameters(self, params: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate PVT parameters"""
        errors = []

        # Check API gravity range
        api = params.get("api_gravity", 0)
        if not (5 <= api <= 55):
            errors.append("API gravity must be between 5 and 55")

        # Check gas specific gravity
        gas_sg = params.get("gas_specific_gravity", 0)
        if not (0.5 <= gas_sg <= 1.5):
            errors.append("Gas specific gravity must be between 0.5 and 1.5")

        # Check viscosities
        viscosities = ["oil_viscosity_cp", "gas_viscosity_cp", "water_viscosity_cp"]
        for visc in viscosities:
            value = params.get(visc, 0)
            if not (0.01 <= value <= 1000):
                errors.append(f"{visc} must be between 0.01 and 1000 cp")

        return len(errors) == 0, errors

    def _validate_geostatistical_parameters(self, params: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate geostatistical parameters"""
        errors = []

        # Check range (correlation length)
        if params.get("range", 0) <= 0:
            errors.append("Range (correlation length) must be positive")

        # Check sill (variance)
        if not (0 < params.get("sill", 0) <= 1):
            errors.append("Sill (variance) must be between 0 and 1")

        # Check anisotropy ratio
        if not (0.1 <= params.get("anisotropy_ratio", 1) <= 10):
            errors.append("Anisotropy ratio must be between 0.1 and 10")

        # Check facies proportions sum
        facies_proportions = params.get("facies_proportions", [])
        if facies_proportions and not np.isclose(sum(facies_proportions), 1.0, atol=1e-3):
            errors.append("Facies proportions must sum to 1.0")

        return len(errors) == 0, errors

    def _validate_well_data(self, well_data: List[Dict[str, Any]]) -> Tuple[bool, List[str]]:
        """Validate well data"""
        errors = []

        for i, well in enumerate(well_data):
            # Check required fields
            if not well.get("name"):
                errors.append(f"Well {i + 1}: Missing well name")

            if well.get("type") not in ["injector", "producer"]:
                errors.append(f"Well {i + 1}: Invalid well type")

            # Check coordinates
            for coord in ["x", "y", "z"]:
                if not isinstance(well.get(coord, 0), (int, float)):
                    errors.append(f"Well {i + 1}: Invalid {coord} coordinate")

        return len(errors) == 0, errors

    def _validate_geomechanics_parameters(
        self, params: Dict[str, Any]
    ) -> Tuple[bool, List[str]]:
        """Validate geomechanics parameters"""
        errors = []

        # Check Young's modulus range (typical reservoir rock: 1e4 to 1e8 psi)
        youngs_modulus = params.get("youngs_modulus", 1e6)
        if not (1e4 <= youngs_modulus <= 1e8):
            errors.append("Young's modulus must be between 1e4 and 1e8 psi")

        # Check Poisson's ratio range (must be between 0 and 0.5)
        poissons_ratio = params.get("poissons_ratio", 0.25)
        if not (0.0 <= poissons_ratio <= 0.5):
            errors.append("Poisson's ratio must be between 0.0 and 0.5")

        # Check Biot coefficient range (must be between 0 and 1)
        biot_coeff = params.get("biot_coefficient", 0.8)
        if not (0.0 <= biot_coeff <= 1.0):
            errors.append("Biot coefficient must be between 0.0 and 1.0")

        # Check rock cohesion (positive value)
        rock_cohesion = params.get("rock_cohesion", 1000.0)
        if rock_cohesion < 0:
            errors.append("Rock cohesion must be non-negative")

        # Check rock friction angle (0-90 degrees)
        friction_angle = params.get("rock_friction_angle", 30.0)
        if not (0.0 <= friction_angle <= 90.0):
            errors.append("Rock friction angle must be between 0 and 90 degrees")

        # Check tensile strength (non-negative)
        tensile_strength = params.get("rock_tensile_strength", 500.0)
        if tensile_strength < 0:
            errors.append("Rock tensile strength must be non-negative")

        # Check horizontal stress ratio (typical range 0.3 to 1.5)
        h_stress_ratio = params.get("initial_horizontal_stress_ratio", 0.7)
        if not (0.1 <= h_stress_ratio <= 2.0):
            errors.append("Horizontal stress ratio must be between 0.1 and 2.0")

        return len(errors) == 0, errors

    def _create_geomechanics_parameters(
        self, data: Dict[str, Any]
    ) -> Optional["GeomechanicsParameters"]:
        """Create GeomechanicsParameters from widget data if geomechanics is enabled"""
        from core.data_models import GeomechanicsParameters

        if not data.get("geomechanics_enabled", False):
            return None

        geomech_data = data.get("geomechanics_parameters", {})
        if geomech_data is None:
            return None

        return GeomechanicsParameters(
            youngs_modulus=geomech_data.get("youngs_modulus", 1.0e6),
            poissons_ratio=geomech_data.get("poissons_ratio", 0.25),
            biot_coefficient=geomech_data.get("biot_coefficient", 0.8),
            rock_cohesion=geomech_data.get("rock_cohesion", 1000.0),
            rock_friction_angle=geomech_data.get("rock_friction_angle", 30.0),
            rock_tensile_strength=geomech_data.get("rock_tensile_strength", 500.0),
            pore_compressibility=geomech_data.get("pore_compressibility", 1e-6),
            permeability_stress_coefficient=geomech_data.get(
                "permeability_stress_coefficient", 1e-5
            ),
            initial_horizontal_stress_ratio=geomech_data.get(
                "initial_horizontal_stress_ratio", 0.7
            ),
        )

    def _generate_heterogeneous_properties(
        self, grid_shape: Tuple[int, int, int], params: GeostatisticalParams
    ) -> Dict[str, np.ndarray]:
        """Generate heterogeneous reservoir properties"""
        nx, ny, nz = grid_shape

        try:
            # Create geostatistical parameters dict
            geo_dict = {
                "variogram_type": params.variogram_type,
                "range": params.range,  # Map to correct field name
                "sill": params.sill,  # Map to correct field name
                "nugget": params.nugget,
                "anisotropy_ratio": params.anisotropy_ratio,
                "anisotropy_angle": params.anisotropy_angle,
                "simulation_method": params.simulation_method,
                "random_seed": params.random_seed,
            }

            # Generate base porosity field
            porosity_field = create_geostatistical_grid((nx, ny), geo_dict)

            # Normalize porosity to realistic range
            porosity_field = 0.05 + 0.30 * porosity_field

            # Generate permeability using porosity-permeability relationship
            porosity_3d = np.stack([porosity_field] * nz, axis=-1)
            perm_x = 100 * (porosity_3d / 0.2) ** 3 * np.exp(5 * porosity_3d)
            perm_y = perm_x / params.anisotropy_ratio
            perm_z = perm_x * 0.1

            return {
                "porosity": porosity_3d,
                "permeability_x": perm_x,
                "permeability_y": perm_y,
                "permeability_z": perm_z,
            }

        except Exception as e:
            logger.error(f"Error generating heterogeneous properties: {e}")
            # Return uniform properties as fallback
            return {
                "porosity": np.full((nx, ny, nz), 0.2),
                "permeability_x": np.full((nx, ny, nz), 100.0),
                "permeability_y": np.full((nx, ny, nz), 100.0),
                "permeability_z": np.full((nx, ny, nz), 10.0),
            }

    def _identify_data_sources(self, widget_data: Dict[str, Any]) -> List[str]:
        """Identify sources of data in widget output"""
        sources = ["data_management_widget"]

        if widget_data.get("reservoir_data", {}).get("imported_from_file"):
            sources.append("imported_reservoir_data")

        if widget_data.get("pvt_properties", {}).get("laboratory_data"):
            sources.append("laboratory_pvt_data")

        if widget_data.get("well_data_list"):
            sources.append("well_data_input")

        return sources

    def _get_processing_steps(self) -> List[str]:
        """Get list of processing steps applied"""
        return [
            "data_extraction",
            "preprocessing",
            "validation",
            "geostatistical_modeling",
            "engine_data_generation",
            "compatibility_testing",
        ]


class DataValidator:
    """Strict data validation for all parameters"""

    @staticmethod
    def validate_range(value: float, min_val: float, max_val: float, name: str) -> Tuple[bool, str]:
        """Validate numeric range"""
        if not (min_val <= value <= max_val):
            return False, f"{name} must be between {min_val} and {max_val}"
        return True, ""

    @staticmethod
    def validate_positive(value: float, name: str) -> Tuple[bool, str]:
        """Validate positive value"""
        if value <= 0:
            return False, f"{name} must be positive"
        return True, ""

    @staticmethod
    def validate_array(array: np.ndarray, name: str, min_length: int = 1) -> Tuple[bool, str]:
        """Validate numpy array"""
        if not isinstance(array, np.ndarray):
            return False, f"{name} must be a numpy array"

        if len(array) < min_length:
            return False, f"{name} must have at least {min_length} elements"

        return True, ""


class UnitConverter:
    """Handle unit conversions for different parameter types"""

    @staticmethod
    def feet_to_meters(feet: float) -> float:
        """Convert feet to meters using PhysicalConstants"""
        return feet * _PHYS_CONSTANTS.FT_TO_M

    @staticmethod
    def meters_to_feet(meters: float) -> float:
        """Convert meters to feet using PhysicalConstants"""
        return meters / _PHYS_CONSTANTS.FT_TO_M

    @staticmethod
    def acres_to_m2(acres: float) -> float:
        """Convert acres to square meters"""
        return acres * 4046.86  # Fixed value for area conversion

    @staticmethod
    def m2_to_acres(m2: float) -> float:
        """Convert square meters to acres"""
        return m2 / 4046.86  # Fixed value for area conversion

    @staticmethod
    def psi_to_pa(psi: float) -> float:
        """Convert psi to Pascal using PhysicalConstants"""
        return psi * _PHYS_CONSTANTS.PSI_TO_PA

    @staticmethod
    def pa_to_psi(pa: float) -> float:
        """Convert Pascal to psi using PhysicalConstants"""
        return pa * _PHYS_CONSTANTS.PA_TO_PSI

    @staticmethod
    def bbl_to_m3(bbl: float) -> float:
        """Convert barrels to cubic meters using PhysicalConstants"""
        return bbl * _PHYS_CONSTANTS.BBLS_TO_M3

    @staticmethod
    def m3_to_bbl(m3: float) -> float:
        """Convert cubic meters to barrels using PhysicalConstants"""
        return m3 * _PHYS_CONSTANTS.M3_TO_BBL


class DataPreprocessor:
    """Preprocess data before engine integration"""

    def __init__(self):
        self.unit_converter = UnitConverter()

    def preprocess_all(self, categorized_data: Dict[str, Any]) -> Dict[str, Any]:
        """Preprocess all data categories"""
        preprocessed = categorized_data.copy()

        # Preprocess each category
        preprocessed["reservoir_parameters"] = self._preprocess_reservoir_parameters(
            categorized_data["reservoir_parameters"]
        )

        preprocessed["pvt_parameters"] = self._preprocess_pvt_parameters(
            categorized_data["pvt_parameters"]
        )

        preprocessed["well_data"] = self._preprocess_well_data(categorized_data["well_data"])

        return preprocessed

    def _preprocess_reservoir_parameters(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Preprocess reservoir parameters"""
        processed = params.copy()

        # Ensure consistent units (convert to SI if needed)
        if "block_sizes" in processed:
            block_sizes = processed["block_sizes"]
            processed["block_sizes_m"] = {
                "dx": self.unit_converter.feet_to_meters(block_sizes.get("dx", 100)),
                "dy": self.unit_converter.feet_to_meters(block_sizes.get("dy", 100)),
                "dz": self.unit_converter.feet_to_meters(block_sizes.get("dz", 20)),
            }

        return processed

    def _preprocess_pvt_parameters(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Preprocess PVT parameters"""
        processed = params.copy()

        # Add derived parameters
        api = processed.get("api_gravity", 35.0)
        gas_sg = processed.get("gas_specific_gravity", 0.65)

        # Estimate oil density from API gravity
        oil_density_ppg = 141.5 / (131.5 + api)  # lb/gal
        processed["oil_density_ppg"] = oil_density_ppg

        # Estimate gas density from specific gravity
        processed["gas_density_ppg"] = gas_sg * 0.0764  # lb/gal

        return processed

    def _preprocess_well_data(self, well_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Preprocess well data"""
        processed_wells = []

        for well in well_data:
            processed_well = well.copy()

            # Ensure coordinate types
            for coord in ["x", "y", "z"]:
                if coord in processed_well:
                    processed_well[coord] = float(processed_well[coord])

            processed_wells.append(processed_well)

        return processed_wells


# Convenience function for easy access
def create_data_integration_engine() -> DataIntegrationEngine:
    """Create and return DataIntegrationEngine instance"""
    return DataIntegrationEngine()
