"""
Fault and Fracture Mechanics Module
Implements Mohr-Coulomb failure criterion, fault slip, and transmissibility changes
for CO₂ storage integrity analysis
"""

import logging
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum

from core.data_models import FaultType, FractureState, FaultProperties, FaultGeometry

logger = logging.getLogger(__name__)



@dataclass
class FaultState:
    """Current state of a fault"""
    fault_id: int
    mechanical_state: FractureState
    cumulative_slip: float  # mm
    current_aperture: float  # mm
    slip_rate: float  # mm/year
    transmissibility_multiplier: float
    last_slip_time: float  # simulation time of last slip event
    healing_rate: float = 0.0  # mm/year - rate of aperture reduction due to healing
    initial_aperture: float = 0.1  # mm - initial aperture
    maximum_aperture: float = 10.0  # mm - maximum aperture
    
    def update_aperture(self, slip_magnitude: float, dilation_angle: float, dt: float):
        """Update fault aperture due to slip and healing"""
        # Dilation during slip
        dilation = slip_magnitude * np.tan(np.radians(dilation_angle))
        self.current_aperture += dilation

        # Healing (aperture reduction over time)
        days_per_year = getattr(self, 'days_per_year', 365.0)
        healing_reduction = self.healing_rate * dt / days_per_year  # Convert to mm/day
        self.current_aperture = max(self.initial_aperture, self.current_aperture - healing_reduction)

        # Clamp to maximum aperture
        self.current_aperture = min(self.current_aperture, self.maximum_aperture)

class MohrCoulombCriterion:
    """
    Implements Mohr-Coulomb failure criterion for fault slip
    τ ≥ c + μσₙ' where σₙ' = σₙ - p
    """
    
    def __init__(self, fault_properties: FaultProperties):
        self.fault_properties = fault_properties
        self.cohesion = fault_properties.cohesion
        self.friction_coefficient = fault_properties.friction_coefficient
        self.friction_angle = fault_properties.calculate_friction_angle()
    
    def calculate_shear_strength(self, effective_normal_stress: float) -> float:
        """Calculate shear strength using Mohr-Coulomb"""
        # Shear strength does not drop below cohesion, even under tension
        return self.cohesion + self.friction_coefficient * max(0, effective_normal_stress)
    
    def check_failure(self, shear_stress: float, effective_normal_stress: float) -> Tuple[bool, float]:
        """Check if failure occurs and return failure ratio"""
        shear_strength = self.calculate_shear_strength(effective_normal_stress)
        failure_occurred = abs(shear_stress) >= shear_strength
        failure_ratio = abs(shear_stress) / shear_strength if shear_strength > 0 else float('inf')
        
        return failure_occurred, failure_ratio
    
    def calculate_slip_magnitude(self, shear_stress: float, effective_normal_stress: float, 
                               stiffness: float) -> float:
        """Calculate slip magnitude using a simple elastic-plastic model"""
        shear_strength = self.calculate_shear_strength(effective_normal_stress)
        
        if shear_stress < shear_strength:
            return 0.0  # No slip
        
        # Excess shear stress beyond strength
        excess_shear = shear_stress - shear_strength
        
        # Simplified slip calculation (would use more complex models in practice)
        slip = excess_shear / stiffness  # mm
        
        return max(0.0, slip)

class FaultStressAnalyzer:
    """Analyzes stress on fault planes and calculates slip potential"""
    
    def __init__(self, grid: Any):
        self.grid = grid
    
    def project_stress_to_fault(self, stress_tensor: np.ndarray, fault: FaultGeometry) -> Tuple[float, float]:
        """
        Project stress tensor onto fault plane to get normal and shear stresses
        """
        normal_vector = fault.calculate_normal_vector()
        slip_direction = fault.calculate_slip_direction()
        
        # Traction vector on fault plane: t = σ · n
        traction = stress_tensor @ normal_vector
        
        # Normal stress component (σₙ)
        normal_stress = np.dot(traction, normal_vector)
        
        # Shear stress component (τ) in slip direction
        shear_stress = np.dot(traction, slip_direction)
        
        return normal_stress, shear_stress
    
    def calculate_effective_stress(self, normal_stress: float, pore_pressure: float, 
                                 biot_coefficient: float) -> float:
        """Calculate effective normal stress using Biot's theory"""
        return normal_stress - biot_coefficient * pore_pressure

class FaultTransmissibilityModel:
    """Models changes in fault transmissibility due to mechanical deformation"""
    
    def __init__(self, initial_transmissibility: float):
        self.initial_transmissibility = initial_transmissibility
    
    def calculate_transmissibility(self, aperture: float, length: float,
                                 fault_type: FaultType) -> float:
        """Calculate fault transmissibility using cubic law for fracture flow"""
        # Cubic law: transmissibility ∝ aperture³
        base_transmissibility = self.initial_transmissibility * (aperture ** 3)

        # Configurable geometry factors for different fault types
        strike_slip_factor = getattr(self, 'strike_slip_geometry_factor', 2.0)
        normal_factor = getattr(self, 'normal_geometry_factor', 1.5)
        reverse_factor = getattr(self, 'reverse_geometry_factor', 1.2)
        oblique_factor = getattr(self, 'oblique_geometry_factor', 1.8)

        # Adjust for fault type and geometry
        if fault_type == FaultType.STRIKE_SLIP:
            # Higher connectivity along strike
            geometry_factor = strike_slip_factor
        elif fault_type == FaultType.NORMAL:
            geometry_factor = normal_factor
        elif fault_type == FaultType.REVERSE:
            geometry_factor = reverse_factor
        else:  # OBLIQUE
            geometry_factor = oblique_factor

        return base_transmissibility * geometry_factor
    
    def calculate_transmissibility_multiplier(self, current_aperture: float,
                                            initial_aperture: float) -> float:
        """Calculate transmissibility multiplier relative to initial state"""
        if initial_aperture <= 0:
            return 1.0

        # ENHANCED: Create continuous evolution without hard clipping
        # Fault transmissibility should evolve gradually as slip accumulates

        # Cubic law: T ∝ aperture³
        aperture_multiplier = (current_aperture / initial_aperture) ** 3

        # FIXED: Increased max enhancement to allow continuous evolution
        # Previous value of 5.0 was causing constant transmissibility after initial slip
        max_enhancement = 50.0  # Allow up to 50x permeability (realistic for major fault zones)
        min_blocking = 0.05  # Severely compressed fault has 5% of matrix permeability

        # Combine effects with softer clipping to allow evolution
        base_multiplier = np.clip(aperture_multiplier, min_blocking, max_enhancement)

        # Add persistence factor: fault never fully recovers matrix permeability
        recovery_factor = 0.8  # Increased from 0.7 for more visible evolution
        final_multiplier = 1.0 + (base_multiplier - 1.0) * recovery_factor

        # Use a softer upper limit that depends on aperture
        # Instead of hard clipping at max_enhancement, use a logarithmic scaling
        # for very high multipliers to prevent unrealistic values
        if final_multiplier > max_enhancement:
            # Logarithmic scaling for extreme values
            final_multiplier = max_enhancement + np.log(final_multiplier - max_enhancement + 1)

        # Clip to reasonable physical limits (much wider range now)
        return np.clip(final_multiplier, min_blocking, 100.0)  # Allow up to 100x for extreme cases

class FaultMechanicsEngine:
    """
    Main fault mechanics engine for CO₂ storage applications
    Handles fault reactivation, slip, and transmissibility changes

    Fault mechanics is opt-in - the engine will only analyze faults when:
    1. Fault data is provided in the grid
    2. The enabled flag is set to True
    """

    def __init__(self, grid: Any, reservoir: Any, enabled: bool = True):
        """
        Initialize fault mechanics engine with reservoir data

        Args:
            grid: Reservoir grid data
            reservoir: ReservoirData object containing faults and properties
            enabled: If False, skip fault mechanics even if fault data exists
        """
        self.grid = grid
        self.reservoir = reservoir
        self.enabled = enabled

        # Extract faults and properties from reservoir
        self.faults = self._extract_faults_from_reservoir()
        self.fault_properties = self._extract_fault_properties_from_reservoir()

        # Initialize analyzers (must be before fault state initialization)
        self.stress_analyzer = FaultStressAnalyzer(grid)
        self.transmissibility_model = FaultTransmissibilityModel(initial_transmissibility=1.0)

        # Mohr-Coulomb criteria for each fault
        self.mohr_coulomb = {}
        for fault_id, props in self.fault_properties.items():
            self.mohr_coulomb[fault_id] = MohrCoulombCriterion(props)

        # Initialize fault states (after transmissibility_model is created)
        self.fault_states = {}
        for fault_id, fault in self.faults.items():
            props = self.fault_properties.get(fault_id)
            if props:
                # ENHANCED: Start with SEVERELY COMPRESSED aperture (strong blocking)
                # The fault is initially highly compressed, creating a flow barrier
                # Even after slip, it never fully recovers due to damage zone effects
                compression_ratio = getattr(self, 'initial_compression_ratio', 0.1)  # Start at 10% of full aperture
                compressed_aperture = props.initial_aperture * compression_ratio

                # Calculate initial transmissibility based on compressed aperture
                initial_multiplier = self.transmissibility_model.calculate_transmissibility_multiplier(
                    compressed_aperture, props.initial_aperture
                )

                self.fault_states[fault_id] = FaultState(
                    fault_id=fault_id,
                    mechanical_state=FractureState.INTACT,
                    cumulative_slip=0.0,
                    current_aperture=compressed_aperture,  # Start COMPRESSED
                    slip_rate=0.0,
                    transmissibility_multiplier=initial_multiplier,  # May be < 1.0 (blocking)
                    last_slip_time=0.0
                )
                logger.info(f"Fault {fault_id} initialized with compressed aperture: {compressed_aperture:.3f} mm "
                          f"(multiplier: {initial_multiplier:.2f}x)")
            else:
                logger.warning(f"No properties found for fault {fault_id}, using defaults")
                compressed_aperture = 0.1 * 0.3  # Compressed from 0.1 mm
                self.fault_states[fault_id] = FaultState(
                    fault_id=fault_id,
                    mechanical_state=FractureState.INTACT,
                    cumulative_slip=0.0,
                    current_aperture=compressed_aperture,
                    slip_rate=0.0,
                    transmissibility_multiplier=0.027,  # (0.03/0.1)³ = 0.027
                    last_slip_time=0.0
                )
        
        # Initialize analyzers
        self.stress_analyzer = FaultStressAnalyzer(grid)
        self.transmissibility_model = FaultTransmissibilityModel(initial_transmissibility=1.0)
        
        # Mohr-Coulomb criteria for each fault
        self.mohr_coulomb = {}
        for fault_id, props in self.fault_properties.items():
            self.mohr_coulomb[fault_id] = MohrCoulombCriterion(props)
    
    def _extract_faults_from_reservoir(self) -> Dict[int, FaultGeometry]:
        """Extract fault geometries from reservoir data"""
        faults = {}

        # Get fault cells from grid data
        if hasattr(self.grid, 'fault_cells') and self.grid.fault_cells:
            for fault_id, fault_cells_list in enumerate(self.grid.fault_cells):
                if fault_cells_list:  # Only create faults that have cells
                    # Create simple fault geometry for linear 1D system
                    n_cells = len(fault_cells_list)
                    if n_cells > 0:
                        # Configurable default fault parameters
                        default_strike = getattr(self, 'default_fault_strike', 90.0)
                        default_dip = getattr(self, 'default_fault_dip', 0.0)
                        default_width = getattr(self, 'default_fault_width', 100.0)

                        # Fault spanning the reservoir
                        faults[fault_id] = FaultGeometry(
                            fault_id=fault_id,
                            fault_type=FaultType.STRIKE_SLIP,  # Simplified for 1D
                            connected_cells=fault_cells_list,
                            strike=default_strike,  # Perpendicular to flow
                            dip=default_dip,      # Horizontal for 1D
                            center_coordinates=np.array([
                                self.grid.depth[fault_cells_list[len(fault_cells_list)//2]],  # Middle depth
                                0.0,  # Center x (ignored in 1D)
                                0.0   # Center y (ignored in 1D)
                            ]),
                            length=np.max(self.grid.depth[fault_cells_list]) - np.min(self.grid.depth[fault_cells_list]),
                            width=default_width,  # Simplified width
                        )

        if not faults:
            logger.info("No faults detected in reservoir data. Fault mechanics analysis disabled.")
            # Return empty faults dict - no synthetic fault creation
            return faults

        return faults
    
    def _extract_fault_properties_from_reservoir(self) -> Dict[int, FaultProperties]:
        """Extract fault properties from reservoir data"""
        fault_props = {}

        # Try to get properties from reservoir.fault_properties (direct attribute)
        if hasattr(self.reservoir, 'fault_properties') and self.reservoir.fault_properties:
            if isinstance(self.reservoir.fault_properties, dict):
                for fault_id, props_data in self.reservoir.fault_properties.items():
                    if isinstance(props_data, FaultProperties):
                        fault_props[fault_id] = props_data
                    elif isinstance(props_data, dict):
                        try:
                            fault_props[fault_id] = FaultProperties(**props_data)
                        except Exception as e:
                            logger.error(f"Failed to create FaultProperties for fault {fault_id}: {e}")

        # ALSO check reservoir.faults dict (ReservoirData stores faults here)
        if not fault_props and hasattr(self.reservoir, 'faults') and self.reservoir.faults:
            if isinstance(self.reservoir.faults, dict) and 'fault_properties' in self.reservoir.faults:
                for fault_id, props_data in self.reservoir.faults['fault_properties'].items():
                    if isinstance(props_data, FaultProperties):
                        fault_props[fault_id] = props_data
                    elif isinstance(props_data, dict):
                        try:
                            fault_props[fault_id] = FaultProperties(**props_data)
                        except Exception as e:
                            logger.error(f"Failed to create FaultProperties for fault {fault_id}: {e}")

        # If no properties found, create defaults for existing faults
        if not fault_props and self.faults:
            logger.info("Creating default fault properties for existing faults")
            for fault_id in self.faults.keys():
                # Configurable default fault properties
                default_cohesion = getattr(self, 'default_fault_cohesion', 500.0)
                default_friction = getattr(self, 'default_fault_friction_coefficient', 0.6)
                default_dilation = getattr(self, 'default_fault_dilation_angle', 5.0)
                default_initial_aperture = getattr(self, 'default_fault_initial_aperture', 0.1)
                default_max_aperture = getattr(self, 'default_fault_maximum_aperture', 10.0)
                default_healing_rate = getattr(self, 'default_fault_healing_rate', 0.01)
                default_stiffness = getattr(self, 'default_fault_stiffness', 1.0e6)

                fault_props[fault_id] = FaultProperties(
                    cohesion=default_cohesion,  # psi
                    friction_coefficient=default_friction,
                    dilation_angle=default_dilation,  # degrees
                    initial_aperture=default_initial_aperture,  # mm
                    maximum_aperture=default_max_aperture,  # mm
                    healing_rate=default_healing_rate,  # mm/year
                    stiffness=default_stiffness  # psi/mm
                )
        
        return fault_props
    
    def analyze_fault_stability(self, stress_state: Any, pore_pressure: np.ndarray,
                              biot_coefficient: float, current_time: float) -> Dict[int, Dict[str, Any]]:
        """
        Analyze stability of all faults under current stress and pressure conditions
        """
        # Early return if fault mechanics is disabled or no faults present
        if not self.enabled or not self.faults:
            return {}

        stability_results = {}

        # Log actual simulation conditions
        logger.debug(f"Fault analysis at time {current_time:.1f}d:")
        logger.debug(f"  Pressure range: {np.min(pore_pressure):.1f} - {np.max(pore_pressure):.1f} psi")
        if hasattr(stress_state, 'stress') and stress_state.stress is not None:
            logger.debug(f"  Stress tensor available with shape: {stress_state.stress.shape}")

        for fault_id, fault in self.faults.items():
            # Get average stress and pressure for fault cells
            fault_cells = fault.connected_cells
            if not fault_cells:
                logger.debug(f"  Fault {fault_id}: No connected cells")
                continue

            # Ensure fault cells are within bounds
            valid_fault_cells = [i for i in fault_cells if i < len(pore_pressure)]
            if not valid_fault_cells:
                logger.debug(f"  Fault {fault_id}: No valid cells within bounds")
                continue

            avg_stress_tensor = self._calculate_average_stress(stress_state, valid_fault_cells)
            avg_pore_pressure = np.mean(pore_pressure[valid_fault_cells])

            # Project stress onto fault plane
            normal_stress, shear_stress = self.stress_analyzer.project_stress_to_fault(
                avg_stress_tensor, fault
            )

            # Calculate effective normal stress
            effective_normal_stress = self.stress_analyzer.calculate_effective_stress(
                normal_stress, avg_pore_pressure, biot_coefficient
            )

            # Check failure using Mohr-Coulomb
            failure_occurred, failure_ratio = self.mohr_coulomb[fault_id].check_failure(
                shear_stress, effective_normal_stress
            )

            # Calculate slip potential
            stiffness = self.mohr_coulomb[fault_id].fault_properties.stiffness
            slip_magnitude = self.mohr_coulomb[fault_id].calculate_slip_magnitude(
                shear_stress, effective_normal_stress, stiffness
            )

            # Update fault state if slip occurred
            if failure_occurred and slip_magnitude > 0:
                self._update_fault_state(fault_id, slip_magnitude, current_time)

            stability_results[fault_id] = {
                'failure_occurred': failure_occurred,
                'failure_ratio': failure_ratio,
                'shear_stress': shear_stress,
                'effective_normal_stress': effective_normal_stress,
                'slip_magnitude': slip_magnitude,
                'normal_stress': normal_stress,
                'pore_pressure': avg_pore_pressure,
                'fault_cells_count': len(valid_fault_cells),
                'connected_cells': valid_fault_cells
            }

            logger.debug(f"  Fault {fault_id}: pressure={avg_pore_pressure:.1f} psi, "
                       f"failure={'YES' if failure_occurred else 'NO'}, "
                       f"ratio={failure_ratio:.2f}, slip={slip_magnitude:.3f} mm")

        return stability_results
    
    def _calculate_average_stress(self, stress_state: Any, cell_indices: List[int]) -> np.ndarray:
        """Calculate average stress tensor over specified cells"""
        n_cells = len(cell_indices)
        avg_stress = np.zeros((3, 3))
        
        for cell_idx in cell_indices:
            stress_tensor = stress_state.get_tensor(cell_idx)
            avg_stress += stress_tensor
        
        return avg_stress / n_cells if n_cells > 0 else avg_stress
    
    def _update_fault_state(self, fault_id: int, slip_magnitude: float, current_time: float):
        """Update fault state after slip event"""
        fault_state = self.fault_states[fault_id]
        fault_props = self.fault_properties[fault_id]
        
        # Update cumulative slip
        fault_state.cumulative_slip += slip_magnitude
        
        # Update aperture due to dilation
        dt = current_time - fault_state.last_slip_time
        fault_state.update_aperture(slip_magnitude, fault_props.dilation_angle, dt)
        
        # Update transmissibility multiplier
        fault_state.transmissibility_multiplier = \
            self.transmissibility_model.calculate_transmissibility_multiplier(
                fault_state.current_aperture, fault_props.initial_aperture
            )
        
        # Update mechanical state
        fault_state.mechanical_state = FractureState.SLIPPING
        fault_state.slip_rate = slip_magnitude / max(dt, 1e-6)  # mm/year
        fault_state.last_slip_time = current_time
        
        logger.info(f"Fault {fault_id} slipped: {slip_magnitude:.3f} mm, "
                   f"aperture: {fault_state.current_aperture:.3f} mm, "
                   f"transmissibility multiplier: {fault_state.transmissibility_multiplier:.2f}")
    
    def get_fault_transmissibility_multipliers(self) -> Dict[int, float]:
        """Get current transmissibility multipliers for all faults"""
        if not self.enabled or not self.faults:
            return {}
        return {fault_id: state.transmissibility_multiplier
                for fault_id, state in self.fault_states.items()}
    
    def calculate_leakage_risk(self, stability_results: Dict[int, Dict[str, Any]],
                             caprock_depth: float) -> Dict[str, Any]:
        """Calculate CO₂ leakage risk through faults"""
        # Early return if fault mechanics is disabled or no results
        if not self.enabled or not self.faults or not stability_results:
            return {
                'leaking_faults': [],
                'high_risk_faults': [],
                'total_leakage_risk': 0.0,
                'normalized_risk': 0.0,
                'risk_level': 'NONE',
                'assessment_depth': 0.0,
                'total_faults': 0
            }

        leaking_faults = []
        high_risk_faults = []

        # Configurable default depth for leakage assessment
        avg_depth = getattr(self, 'default_leakage_depth', 5000.0)  # Default depth
        if hasattr(self.grid, 'depth') and len(self.grid.depth) > 0:
            avg_depth = np.mean(self.grid.depth)
        elif hasattr(self.reservoir, 'average_depth'):
            avg_depth = self.reservoir.average_depth

        logger.debug(f"Leakage risk assessment with avg depth: {avg_depth:.1f} ft, caprock: {caprock_depth:.1f} ft")

        for fault_id, result in stability_results.items():
            if fault_id not in self.fault_states:
                logger.debug(f"Fault {fault_id} not in states, skipping")
                continue

            fault_state = self.fault_states[fault_id]

            # Configurable leakage criteria
            transmissibility_threshold = getattr(self, 'leakage_transmissibility_threshold', 5.0)
            baseline_pressure = getattr(self, 'baseline_pressure', 2000.0)
            failure_threshold = getattr(self, 'leakage_failure_threshold', 0.6)
            pressure_factor_threshold = getattr(self, 'leakage_pressure_factor_threshold', 1.2)

            is_slipping = fault_state.mechanical_state == FractureState.SLIPPING
            high_transmissibility = fault_state.transmissibility_multiplier > transmissibility_threshold
            shallow_fault = avg_depth < caprock_depth

            # Dynamic leakage criteria based on simulation conditions
            pressure_factor = result.get('pore_pressure', baseline_pressure) / baseline_pressure  # Normalized to baseline
            failure_severity = result.get('failure_ratio', 0.0)

            # Enhanced leakage criteria
            if is_slipping and high_transmissibility and shallow_fault:
                leaking_faults.append(fault_id)
                logger.debug(f"Fault {fault_id}: ACTIVE LEAKAGE (slipping + high transmissibility + shallow)")
            elif failure_severity > failure_threshold and pressure_factor > pressure_factor_threshold:
                high_risk_faults.append(fault_id)
                logger.debug(f"Fault {fault_id}: HIGH RISK (failure={failure_severity:.2f}, pressure={pressure_factor:.2f})")

        total_leakage_risk = len(leaking_faults) + 0.5 * len(high_risk_faults)
        max_risk = max(1, len(self.faults))  # Avoid division by zero
        normalized_risk = total_leakage_risk / max_risk

        risk_level = self._classify_risk_level(normalized_risk)
        logger.info(f"Leakage assessment: {len(leaking_faults)} leaking, {len(high_risk_faults)} high risk, level={risk_level}")

        return {
            'leaking_faults': leaking_faults,
            'high_risk_faults': high_risk_faults,
            'total_leakage_risk': total_leakage_risk,
            'normalized_risk': normalized_risk,
            'risk_level': risk_level,
            'assessment_depth': avg_depth,
            'total_faults': max_risk
        }
    
    def _classify_risk_level(self, normalized_risk: float) -> str:
        """Classify leakage risk level"""
        # Configurable risk thresholds
        low_threshold = getattr(self, 'low_risk_threshold', 0.1)
        moderate_threshold = getattr(self, 'moderate_risk_threshold', 0.3)
        high_threshold = getattr(self, 'high_risk_threshold', 0.6)

        if normalized_risk < low_threshold:
            return "LOW"
        elif normalized_risk < moderate_threshold:
            return "MODERATE"
        elif normalized_risk < high_threshold:
            return "HIGH"
        else:
            return "VERY_HIGH"


def analyze_injection_induced_slip(injection_pressure: float, initial_pressure: float,
                                 fault_mechanics: FaultMechanicsEngine, stress_state: Any,
                                 biot_coefficient: float) -> Dict[str, Any]:
    """
    Analyze fault slip risk due to CO₂ injection pressure increase
    """
    pressure_increase = injection_pressure - initial_pressure
    
    # Simplified analysis: calculate critical pressure for fault reactivation
    critical_pressures = {}
    for fault_id, fault in fault_mechanics.faults.items():
        # Get fault properties
        props = fault_mechanics.fault_properties[fault_id]
        mc_criterion = MohrCoulombCriterion(props)
        
        # Calculate initial stress state on fault
        fault_cells = fault.connected_cells
        if not fault_cells:
            continue
            
        avg_stress_tensor = fault_mechanics._calculate_average_stress(stress_state, fault_cells)
        normal_stress, shear_stress = fault_mechanics.stress_analyzer.project_stress_to_fault(
            avg_stress_tensor, fault
        )
        
        # Calculate critical pressure for slip
        # From Mohr-Coulomb: τ = c + μ(σₙ - αp_crit)
        # So: p_crit = (σₙ - (τ - c)/μ) / α
        critical_pressure = (normal_stress - (shear_stress - props.cohesion) / props.friction_coefficient) / biot_coefficient
        
        safety_margin_factor = getattr(fault_mechanics, 'injection_safety_margin', 0.8)
        critical_pressures[fault_id] = {
            'critical_pressure': critical_pressure,
            'pressure_margin': critical_pressure - injection_pressure,
            'safe_injection_pressure': critical_pressure * safety_margin_factor
        }
    
    return critical_pressures