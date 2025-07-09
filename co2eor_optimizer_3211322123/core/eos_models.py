import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union

import numpy as np
from scipy.optimize import minimize
import pandas as pd

# Import the thermo library for advanced thermodynamic calculations
try:
    import thermo
    from thermo.eos_mix import PRMIX, SRKMIX
    from thermo.flash import FlashVLN
    from thermo.viscosity import Lohrenz_Bray_Clark
    from thermo.utils import F_to_K, psia_to_Pa, R_to_K, atm_to_Pa
    THERMO_AVAILABLE = True
except ImportError:
    logging.critical("The 'thermo' library is not installed. EOS models cannot function. "
                     "Please install it using: pip install thermo")
    THERMO_AVAILABLE = False
    # Define dummy classes/functions to prevent import errors in other modules
    class DummyThermo: pass
    PRMIX, SRKMIX, FlashVLN, Lohrenz_Bray_Clark = DummyThermo, DummyThermo, DummyThermo, DummyThermo
    def F_to_K(v): return (v - 32) * 5/9 + 273.15
    def psia_to_Pa(v): return v * 6894.76
    def R_to_K(v): return v * 5/9
    def atm_to_Pa(v): return v * 101325.0 # <<< --- FIXED: Added missing dummy function

try:
    from .data_models import EOSModelParameters
except ImportError:
    class EOSModelParameters:
        """Dummy EOSModelParameters to allow module import if data_models is not found."""
        def __init__(self, eos_type: str, component_properties: np.ndarray, binary_interaction_coeffs: np.ndarray):
            self.eos_type = eos_type
            self.component_properties = component_properties
            self.binary_interaction_coeffs = binary_interaction_coeffs
    logging.critical("eos_models: Could not import EOSModelParameters. Using a dummy class.")

logger = logging.getLogger(__name__)

# Standard conditions for FVF calculations
P_STD_PA = atm_to_Pa(1)
T_STD_K = F_to_K(60)

class EOSModel(ABC):
    """Abstract base class for Equation of State models, integrated with the 'thermo' library."""
    def __init__(self, eos_params: EOSModelParameters):
        if not THERMO_AVAILABLE:
            raise ImportError("The 'thermo' library is required for EOSModel functionality.")
        if not isinstance(eos_params, EOSModelParameters):
            raise TypeError("eos_params must be an instance of EOSModelParameters")

        self.params = eos_params
        
        # Unpack component properties from the numpy array
        # Expected columns: Name, Mol Frac, MW, Tc(R), Pc(psia), Omega, Vol Shift
        names_in = self.params.component_properties[:, 0]
        self.zi = self.params.component_properties[:, 1].astype(float)
        MWs_in = self.params.component_properties[:, 2].astype(float)
        Tcs_in_R = self.params.component_properties[:, 3].astype(float)
        Pcs_in_psia = self.params.component_properties[:, 4].astype(float)
        omegas_in = self.params.component_properties[:, 5].astype(float)
        # Volume shift parameter (s_i) for Peneloux correction is column 6
        vol_shifts_in = self.params.component_properties[:, 6].astype(float)

        # Initialize lists for fully populated, SI-unit properties
        self.names: List[str] = list(names_in)
        self.MWs: List[float] = []
        self.Tcs: List[float] = []
        self.Pcs: List[float] = []
        self.omegas: List[float] = []
        self.Vcs: List[float] = []
        self.s_V: List[float] = list(vol_shifts_in)

        # Auto-populate missing properties using thermo.Chemical
        for i, name in enumerate(self.names):
            chem = thermo.Chemical(name)
            
            # Molecular Weight
            mw = MWs_in[i] if MWs_in[i] > 0 else chem.MW
            if mw is None: raise ValueError(f"Could not determine Molecular Weight for {name}")
            self.MWs.append(mw)
            
            # Critical Temperature
            tc = R_to_K(Tcs_in_R[i]) if Tcs_in_R[i] > 0 else chem.Tc
            if tc is None: raise ValueError(f"Could not determine Tc for {name}")
            self.Tcs.append(tc)

            # Critical Pressure
            pc = psia_to_Pa(Pcs_in_psia[i]) if Pcs_in_psia[i] > 0 else chem.Pc
            if pc is None: raise ValueError(f"Could not determine Pc for {name}")
            self.Pcs.append(pc)
            
            # Acentric Factor
            omega = omegas_in[i] if not np.isnan(omegas_in[i]) else chem.omega
            if omega is None: raise ValueError(f"Could not determine Acentric Factor (omega) for {name}")
            self.omegas.append(omega)
            
            # Critical Volume (required for viscosity and density corrections)
            vc = chem.Vc
            if vc is None: raise ValueError(f"Could not determine Critical Volume (Vc) for {name}")
            self.Vcs.append(vc)

        self.kijs = self.params.binary_interaction_coeffs
        self.eos_mixture: Optional[Union[PRMIX, SRKMIX]] = None

    @abstractmethod
    def calculate_properties(self, pressure_psia: float, temperature_F: float) -> Dict[str, Any]:
        """
        Calculates fluid properties at a given pressure and temperature.
        This must be implemented by subclasses (PengRobinsonEOS, SoaveRedlichKwongEOS).
        """
        pass

    def _validate_inputs(self, pressure_psia: float, temperature_F: float):
        if not (isinstance(pressure_psia, (int, float)) and pressure_psia > 0):
            raise ValueError("Pressure must be a positive number.")
        if not (isinstance(temperature_F, (int, float)) and temperature_F > -459.67):
            raise ValueError("Temperature must be a valid number in Fahrenheit.")

    def _perform_flash_and_calc_props(self, T: float, P: float) -> Dict[str, Any]:
        """
        Core calculation logic using a FlashVLN object.
        Args:
            T (float): Temperature in Kelvin.
            P (float): Pressure in Pascals.
        Returns:
            A dictionary of calculated fluid properties.
        """
        if self.eos_mixture is None:
            raise RuntimeError("EOS mixture model has not been initialized in the subclass.")

        flash_obj = FlashVLN(self.Tcs, self.Pcs, self.omegas, self.kijs, self.zi)
        flash_obj.flash(T=T, P=P)

        results: Dict[str, Any] = {'phase': flash_obj.phase, 'status': 'Success'}
        
        # --- Flash at Standard Conditions for FVF ---
        try:
            # Only flash the liquid part (xs) at standard conditions to get stock tank density
            if flash_obj.phase in ('L', 'L-V') and flash_obj.xs is not None and sum(flash_obj.xs) > 1e-6:
                flash_std_liquid = FlashVLN(self.Tcs, self.Pcs, self.omegas, self.kijs, flash_obj.xs)
                flash_std_liquid.flash(T=T_STD_K, P=P_STD_PA)
                rho_l_std_molar = flash_std_liquid.rho_l
            else:
                 rho_l_std_molar = None
        except (ValueError, RuntimeError):
             rho_l_std_molar = None # Could not flash at std conditions

        # Single Liquid Phase
        if flash_obj.phase == 'L':
            results['oil_viscosity_cp'] = self.eos_mixture.viscosity_liquid(T, P, self.zi, self.MWs, self.Vcs) * 1000.0
            results['oil_density_kg_m3'] = self.eos_mixture.rho_l
            results['z_factor_liquid'] = self.eos_mixture.Z_l
            results['oil_fvf_rb_stb'] = (rho_l_std_molar / self.eos_mixture.rho_l) if rho_l_std_molar else None
            results['vapor_mole_fraction'] = 0.0

        # Single Gas Phase
        elif flash_obj.phase == 'V':
            results['gas_viscosity_cp'] = self.eos_mixture.viscosity_gas(T, P, self.zi, self.MWs, self.Vcs) * 1000.0
            results['gas_density_kg_m3'] = self.eos_mixture.rho_g
            results['z_factor_gas'] = self.eos_mixture.Z_g
            results['gas_fvf_rcf_scf'] = (P_STD_PA / T_STD_K) * (self.eos_mixture.Z_g * T / P)
            results['vapor_mole_fraction'] = 1.0
            
        # Two-Phase (Liquid-Vapor)
        elif flash_obj.phase == 'L-V':
            results['vapor_mole_fraction'] = flash_obj.beta
            # Liquid phase properties
            results['oil_viscosity_cp'] = self.eos_mixture.viscosity_liquid(T, P, flash_obj.xs, self.MWs, self.Vcs) * 1000.0
            results['oil_density_kg_m3'] = flash_obj.rho_l
            results['z_factor_liquid'] = flash_obj.Z_l
            results['oil_fvf_rb_stb'] = (rho_l_std_molar / flash_obj.rho_l) if rho_l_std_molar else None
            # Gas phase properties
            results['gas_viscosity_cp'] = self.eos_mixture.viscosity_gas(T, P, flash_obj.ys, self.MWs, self.Vcs) * 1000.0
            results['gas_density_kg_m3'] = flash_obj.rho_g
            results['z_factor_gas'] = flash_obj.Z_g
            results['gas_fvf_rcf_scf'] = (P_STD_PA / T_STD_K) * (flash_obj.Z_g * T / P)

        else:
            results['status'] = f'Flash calculation resulted in an unexpected phase: {flash_obj.phase}'
            logger.error(results['status'])

        return results

class PengRobinsonEOS(EOSModel):
    """Implementation of the Peng-Robinson EOS."""
    def __init__(self, eos_params: EOSModelParameters):
        super().__init__(eos_params)
        self.eos_mixture = PRMIX(Tcs=self.Tcs, Pcs=self.Pcs, omegas=self.omegas, kijs=self.kijs, zs=self.zi, Vcs=self.Vcs, MWs=self.MWs, s_V=self.s_V)

    def calculate_properties(self, pressure_psia: float, temperature_F: float) -> Dict[str, Any]:
        self._validate_inputs(pressure_psia, temperature_F)
        logger.info(f"Calculating properties with Peng-Robinson EOS for {len(self.names)} components.")
        
        try:
            T_K = F_to_K(temperature_F)
            P_Pa = psia_to_Pa(pressure_psia)
            return self._perform_flash_and_calc_props(T_K, P_Pa)
        except Exception as e:
            logger.error(f"Peng-Robinson calculation failed: {e}", exc_info=True)
            return {'status': f'Error: {e}'}

class SoaveRedlichKwongEOS(EOSModel):
    """Implementation of the Soave-Redlich-Kwong EOS."""
    def __init__(self, eos_params: EOSModelParameters):
        super().__init__(eos_params)
        self.eos_mixture = SRKMIX(Tcs=self.Tcs, Pcs=self.Pcs, omegas=self.omegas, kijs=self.kijs, zs=self.zi, Vcs=self.Vcs, MWs=self.MWs, s_V=self.s_V)

    def calculate_properties(self, pressure_psia: float, temperature_F: float) -> Dict[str, Any]:
        self._validate_inputs(pressure_psia, temperature_F)
        logger.info(f"Calculating properties with Soave-Redlich-Kwong EOS for {len(self.names)} components.")
        
        try:
            T_K = F_to_K(temperature_F)
            P_Pa = psia_to_Pa(pressure_psia)
            return self._perform_flash_and_calc_props(T_K, P_Pa)
        except Exception as e:
            logger.error(f"Soave-Redlich-Kwong calculation failed: {e}", exc_info=True)
            return {'status': f'Error: {e}'}

def fit_kij_to_pvt(
    eos_class: Union[type[PengRobinsonEOS], type[SoaveRedlichKwongEOS]],
    base_eos_params: EOSModelParameters,
    experimental_data: pd.DataFrame,
    initial_kij_guess: Optional[np.ndarray] = None,
    target_property: str = 'oil_density_kg_m3'
) -> Dict[str, Any]:
    """
    Fits binary interaction parameters (kij) to match experimental PVT data.

    Args:
        eos_class: The EOS model class to use (PengRobinsonEOS or SoaveRedlichKwongEOS).
        base_eos_params: The base EOS parameters, including components and mole fractions.
        experimental_data: A pandas DataFrame with columns 'pressure_psia', 'temperature_F',
                           and a column matching `target_property` (e.g., 'oil_density_kg_m3').
        initial_kij_guess: An optional initial guess for the kij matrix. If None, assumes zeros.
        target_property: The property in the calculated results to match against experimental data.

    Returns:
        A dictionary containing the optimized kij matrix and the final error score.
    """
    num_components = len(base_eos_params.component_properties)
    if initial_kij_guess is None:
        initial_kij_guess = np.zeros((num_components, num_components))
    
    # We only need to optimize the upper triangle of the kij matrix
    upper_triangle_indices = np.triu_indices(num_components, k=1)
    initial_guess_flat = initial_kij_guess[upper_triangle_indices]

    def objective_function(kij_flat: np.ndarray) -> float:
        # Reconstruct the symmetric kij matrix from the flattened upper triangle
        kij_matrix = np.zeros((num_components, num_components))
        kij_matrix[upper_triangle_indices] = kij_flat
        kij_matrix = kij_matrix + kij_matrix.T

        # Create a temporary EOSModelParameters instance with the new kijs
        temp_eos_params = EOSModelParameters(
            eos_type=base_eos_params.eos_type,
            component_properties=base_eos_params.component_properties,
            binary_interaction_coeffs=kij_matrix
        )
        
        # Instantiate the EOS model
        try:
            model = eos_class(temp_eos_params)
        except Exception as e:
            logger.error(f"Failed to instantiate EOS model during optimization: {e}")
            return 1e12 # Return a large error

        total_squared_error = 0.0
        
        for _, row in experimental_data.iterrows():
            pressure = row['pressure_psia']
            temperature = row['temperature_F']
            experimental_value = row[target_property]
            
            calc_props = model.calculate_properties(pressure, temperature)
            
            if 'status' in calc_props and 'Error' in calc_props['status']:
                total_squared_error += 1e6 # Penalize failed calculations
                continue

            calculated_value = calc_props.get(target_property)
            if calculated_value is None:
                total_squared_error += 1e6 # Penalize if target property is not calculated
                continue
            
            error = ((calculated_value - experimental_value) / experimental_value)**2
            total_squared_error += error
            
        return total_squared_error

    logger.info(f"Starting kij fitting for {eos_class.__name__} against '{target_property}'...")
    result = minimize(
        objective_function,
        initial_guess_flat,
        method='Nelder-Mead', # A robust method for non-smooth functions
        options={'xatol': 1e-4, 'fatol': 1e-4, 'disp': True}
    )

    if result.success:
        logger.info(f"Kij fitting successful. Final function value (error score): {result.fun:.6f}")
        # Reconstruct the final optimized matrix
        optimized_kij_matrix = np.zeros((num_components, num_components))
        optimized_kij_matrix[upper_triangle_indices] = result.x
        optimized_kij_matrix = optimized_kij_matrix + optimized_kij_matrix.T
        return {
            'optimized_kij_matrix': optimized_kij_matrix,
            'final_error': result.fun,
            'success': True,
            'message': result.message
        }
    else:
        logger.error(f"Kij fitting failed: {result.message}")
        return {
            'optimized_kij_matrix': None,
            'final_error': result.fun,
            'success': False,
            'message': result.message
        }