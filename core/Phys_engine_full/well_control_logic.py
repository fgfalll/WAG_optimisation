
"""
Sophisticated Well Control Logic System for CO₂-CCUS Operations
Implements dynamic state machine with multi-variable control and automated transitions
"""

import logging
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field
import numpy as np
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class WellControlState(Enum):
    """Primary operational modes for well control state machine"""
    INJECTOR_CO2 = "injector_co2"           # Actively injecting supercritical CO2
    PRODUCER = "producer"                   # Actively producing reservoir fluids
    OBSERVER = "observer"                   # Shut-in, monitoring BHP and temperature
    SHUT_IN_SAFETY = "shut_in_safety"       # Emergency shutdown triggered by safety protocols
    SHUT_IN_MAINTENANCE = "shut_in_maintenance"  # Planned shutdown for well servicing
    CLEANUP = "cleanup"                     # Post-injection/production phase for wellbore conditioning
    TRANSITIONING = "transitioning"         # Temporary state during mode changes

class ControlMode(Enum):
    """Available control modes for well operations"""
    BHP = "bhp"                    # Bottom-Hole Pressure control
    SURFACE_RATE = "surface_rate"  # Surface rate control (mass or volumetric)
    SURFACE_PRESSURE = "surface_pressure"  # Surface pressure control
    RESERVOIR_RATE = "reservoir_rate"  # Reservoir rate control

class SafetyEvent(Enum):
    """Types of safety events that trigger emergency shutdowns"""
    BHP_EXCEEDED = "bhp_exceeded"
    BHP_BELOW_MIN = "bhp_below_min"
    SURFACE_PRESSURE_EXCEEDED = "surface_pressure_exceeded"
    TEMPERATURE_DEVIATION = "temperature_deviation"
    SEISMIC_EVENT = "seismic_event"
    CO2_LEAK_DETECTED = "co2_leak_detected"
    EQUIPMENT_FAILURE = "equipment_failure"

@dataclass
class WellControlParameters:
    """Control and constraint parameters for well operations"""
    
    # Well identification
    well_name: str
    well_type: WellControlState
    
    # Control parameters
    control_mode: ControlMode = ControlMode.BHP
    control_setpoint: float = 0.0  # Current control target
    
    # Operational constraints
    bhp_max: float = 12000.0       # Maximum allowable BHP (psi) - Will be calculated dynamically
    bhp_min: float = 1000.0        # Minimum allowable BHP (psi)
    fracture_pressure_psi: float = 0.0  # Calculated fracture pressure (psi) - Set dynamically
    depth_ft: float = 8000.0       # Well depth (ft) - Used for fracture pressure calculation
    fracture_gradient: float = 1.0  # Fracture gradient (psi/ft) - Typical 0.8-1.2
    surface_rate_max: float = 100000.0  # Maximum surface rate (Mscf/day for gas, bbl/day for liquid)
    surface_pressure_max: float = 5000.0  # Maximum surface pressure (psi)
    temperature_max: float = 300.0  # Maximum downhole temperature (°F)
    temperature_min: float = 100.0  # Minimum downhole temperature (°F)
    co2_purity_threshold: float = 0.95  # Minimum CO2 purity for injection stream
    flowline_temperature_min: float = 50.0  # Minimum flowline temperature (°F)
    
    # Safety thresholds
    seismic_magnitude_threshold: float = 2.5  # Magnitude threshold for seismic shutdown
    surface_co2_concentration_threshold: float = 5000.0  # ppm threshold for leak detection
    
    # Operational targets
    total_injected_mass_target: float = 0.0  # Target total CO2 mass for injection wells (tonne)
    pressure_buildup_duration: float = 24.0  # Duration for pressure buildup tests (hours)
    
    # Equipment health monitoring
    valve_actuation_time_threshold: float = 5.0  # Maximum acceptable valve actuation time (seconds)
    corrosion_rate_threshold: float = 0.01  # Maximum acceptable corrosion rate (mm/year)
    
    # Reservoir management parameters
    plume_steering_enabled: bool = False
    capacity_optimization_enabled: bool = False
    containment_boundary_distance: float = 100.0  # Distance to containment boundary (ft)
    
    def calculate_fracture_pressure(self, reservoir_pressure: float = 4500.0) -> float:
        """Calculate realistic fracture pressure based on depth and stress conditions"""
        if self.fracture_pressure_psi > 0:
            return self.fracture_pressure_psi

        # Calculate fracture pressure using multiple methods
        # Method 1: Depth-based calculation
        depth_fracture = self.depth_ft * self.fracture_gradient

        # Method 2: Reservoir pressure + overburden stress factor
        stress_factor = 1.5  # Typical overburden stress factor
        pressure_fracture = reservoir_pressure * stress_factor

        # Use the more conservative (lower) value
        calculated_fracture = min(depth_fracture, pressure_fracture)

        # Apply safety factor
        self.fracture_pressure_psi = calculated_fracture * 0.9  # 90% of calculated value

        logger.info(f"Calculated fracture pressure: {self.fracture_pressure_psi:.0f} psi "
                   f"(depth: {depth_fracture:.0f}, pressure: {pressure_fracture:.0f})")

        return self.fracture_pressure_psi

    def validate_constraints(self) -> bool:
        """Validate that constraint parameters are physically reasonable"""
        # Calculate fracture pressure if not already set
        self.calculate_fracture_pressure()

        # Update BHP max to be below fracture pressure
        self.bhp_max = min(self.bhp_max, self.fracture_pressure_psi)

        if self.bhp_max <= self.bhp_min:
            logger.error(f"Well {self.well_name}: BHP max ({self.bhp_max}) must be greater than BHP min ({self.bhp_min})")
            return False
        
        if self.temperature_max <= self.temperature_min:
            logger.error(f"Well {self.well_name}: Temperature max ({self.temperature_max}) must be greater than temperature min ({self.temperature_min})")
            return False
        
        if self.surface_rate_max <= 0:
            logger.error(f"Well {self.well_name}: Surface rate max must be positive")
            return False
            
        if self.surface_pressure_max <= 0:
            logger.error(f"Well {self.well_name}: Surface pressure max must be positive")
            return False
            
        return True

@dataclass
class WellSensorData:
    """Real-time sensor data from well monitoring"""
    timestamp: datetime
    bottom_hole_pressure: float  # psi
    bottom_hole_temperature: float  # °F
    surface_pressure: float  # psi
    surface_temperature: float  # °F
    injection_rate: Optional[float] = None  # Mscf/day for gas, bbl/day for liquid
    production_rate: Optional[float] = None  # bbl/day
    co2_purity: Optional[float] = None  # Fraction (0-1)
    seismic_magnitude: Optional[float] = None  # Local seismic event magnitude
    surface_co2_concentration: Optional[float] = None  # ppm
    valve_actuation_time: Optional[float] = None  # seconds
    corrosion_rate: Optional[float] = None  # mm/year
    
    def is_valid(self) -> bool:
        """Check if sensor data contains valid measurements"""
        return (self.bottom_hole_pressure > 0 and 
                self.bottom_hole_temperature > 0 and 
                self.surface_pressure >= 0)

@dataclass
class WellControlStateTransition:
    """Record of state transitions with reasons"""
    timestamp: datetime
    from_state: WellControlState
    to_state: WellControlState
    reason: str
    sensor_data: Optional[WellSensorData] = None
    triggered_by_safety: bool = False

class WellControlLogic:
    """
    Sophisticated well control logic with dynamic state machine and multi-variable control
    """
    
    def __init__(self):
        self.wells: Dict[str, Dict[str, Any]] = {}  # well_name -> {params, state, transitions, etc.}
        self.multi_well_optimization_enabled = False
        
    def add_well(self, control_params: WellControlParameters):
        """Add a well to the control system"""
        if control_params.well_name in self.wells:
            raise ValueError(f"Well {control_params.well_name} already exists in control system")
        
        if not control_params.validate_constraints():
            raise ValueError(f"Invalid control parameters for well {control_params.well_name}")
        
        self.wells[control_params.well_name] = {
            'params': control_params,
            'current_state': WellControlState.OBSERVER,  # Default initial state
            'previous_state': None,
            'state_transitions': [],
            'total_injected_mass': 0.0,
            'shutdown_time': None,
            'equipment_health_score': 1.0,
            'control_history': []
        }
    
    def update_sensor_data(self, well_name: str, sensor_data: WellSensorData) -> WellControlState:
        """
        Update well state based on real-time sensor data
        Returns the new state after processing
        """
        if well_name not in self.wells:
            raise ValueError(f"Well {well_name} not found in control system")
        
        well_data = self.wells[well_name]
        
        if not sensor_data.is_valid():
            logger.warning(f"Well {well_name}: Invalid sensor data received")
            return well_data['current_state']
        
        # Check for safety triggers first (highest priority)
        safety_event = self._check_safety_triggers(well_name, sensor_data)
        if safety_event:
            return self._handle_safety_event(well_name, safety_event, sensor_data)
        
        # Check for operational triggers based on current state
        new_state = self._check_operational_triggers(well_name, sensor_data)
        
        if new_state != well_data['current_state']:
            self._transition_state(well_name, new_state, "Operational trigger", sensor_data)
        
        # Update equipment health monitoring
        self._update_equipment_health(well_name, sensor_data)
        
        # Store control history
        self._record_control_history(well_name, sensor_data)
        
        return well_data['current_state']
    
    def _check_safety_triggers(self, well_name: str, sensor_data: WellSensorData) -> Optional[SafetyEvent]:
        """Check for conditions that require immediate safety shutdown"""
        well_data = self.wells[well_name]
        params = well_data['params']
        current_state = well_data['current_state']

        if current_state in [WellControlState.SHUT_IN_SAFETY, WellControlState.SHUT_IN_MAINTENANCE]:
            return None # Already shut in

        # BHP constraints
        if (params.well_type == WellControlState.INJECTOR_CO2 and
            sensor_data.bottom_hole_pressure > params.bhp_max):
            return SafetyEvent.BHP_EXCEEDED
            
        if (params.well_type == WellControlState.PRODUCER and
            sensor_data.bottom_hole_pressure < params.bhp_min):
            return SafetyEvent.BHP_BELOW_MIN
        
        # Surface pressure constraint
        if sensor_data.surface_pressure > params.surface_pressure_max:
            return SafetyEvent.SURFACE_PRESSURE_EXCEEDED
        
        # Temperature deviation
        if (sensor_data.bottom_hole_temperature > params.temperature_max or
            sensor_data.bottom_hole_temperature < params.temperature_min):
            return SafetyEvent.TEMPERATURE_DEVIATION
        
        # Seismic event
        if (sensor_data.seismic_magnitude is not None and
            sensor_data.seismic_magnitude > params.seismic_magnitude_threshold):
            return SafetyEvent.SEISMIC_EVENT
        
        # CO2 leak detection
        if (sensor_data.surface_co2_concentration is not None and
            sensor_data.surface_co2_concentration > params.surface_co2_concentration_threshold):
            return SafetyEvent.CO2_LEAK_DETECTED
        
        # Equipment failure indicators
        if (sensor_data.valve_actuation_time is not None and
            sensor_data.valve_actuation_time > params.valve_actuation_time_threshold):
            return SafetyEvent.EQUIPMENT_FAILURE
            
        return None
    
    def _handle_safety_event(self, well_name: str, safety_event: SafetyEvent, sensor_data: WellSensorData) -> WellControlState:
        """Handle safety events with immediate shutdown"""
        well_data = self.wells[well_name]
        
        reason_map = {
            SafetyEvent.BHP_EXCEEDED: "BHP exceeded maximum allowable limit",
            SafetyEvent.BHP_BELOW_MIN: "BHP fell below minimum allowable limit",
            SafetyEvent.SURFACE_PRESSURE_EXCEEDED: "Surface pressure exceeded maximum limit",
            SafetyEvent.TEMPERATURE_DEVIATION: "Temperature outside safe operating envelope",
            SafetyEvent.SEISMIC_EVENT: "Seismic event detected above threshold",
            SafetyEvent.CO2_LEAK_DETECTED: "Surface CO2 concentration leak detected",
            SafetyEvent.EQUIPMENT_FAILURE: "Equipment failure indicated"
        }
        
        reason = reason_map.get(safety_event, "Unknown safety event")
        logger.warning(f"Well {well_name}: Safety shutdown triggered - {reason}")
        
        well_data['shutdown_time'] = sensor_data.timestamp
        return self._transition_state(well_name, WellControlState.SHUT_IN_SAFETY, reason, sensor_data, True)
    
    def _check_operational_triggers(self, well_name: str, sensor_data: WellSensorData) -> WellControlState:
        """Check for operational state transitions"""
        well_data = self.wells[well_name]
        params = well_data['params']
        current_state = well_data['current_state']
        
        if current_state == WellControlState.OBSERVER:
            if params.well_type == WellControlState.INJECTOR_CO2 and sensor_data.injection_rate is not None and sensor_data.injection_rate > 0:
                return WellControlState.INJECTOR_CO2
            if params.well_type == WellControlState.PRODUCER and sensor_data.production_rate is not None and sensor_data.production_rate > 0:
                return WellControlState.PRODUCER

        if current_state == WellControlState.INJECTOR_CO2:
            # Check for injection completion
            if (params.total_injected_mass_target > 0 and
                well_data['total_injected_mass'] >= params.total_injected_mass_target):
                return WellControlState.CLEANUP
            
            # Check for CO2 purity issues
            if (sensor_data.co2_purity is not None and
                sensor_data.co2_purity < params.co2_purity_threshold):
                logger.warning(f"Well {well_name}: CO2 purity below threshold")
                return WellControlState.SHUT_IN_MAINTENANCE
        
        elif current_state == WellControlState.PRODUCER:
            # Check for flowline temperature issues
            if (sensor_data.surface_temperature < params.flowline_temperature_min):
                logger.warning(f"Well {well_name}: Flowline temperature below minimum")
                return WellControlState.SHUT_IN_MAINTENANCE
        
        elif current_state == WellControlState.SHUT_IN_SAFETY:
            # Safety shutdown recovery logic
            if self._can_recover_from_safety_shutdown(well_name, sensor_data):
                return WellControlState.OBSERVER
        
        elif current_state == WellControlState.CLEANUP:
            # Cleanup completion logic
            if self._is_cleanup_complete(well_name, sensor_data):
                return WellControlState.OBSERVER
        
        return current_state  # No state change
    
    def _can_recover_from_safety_shutdown(self, well_name: str, sensor_data: WellSensorData) -> bool:
        """Determine if safe to recover from safety shutdown"""
        well_data = self.wells[well_name]
        params = well_data['params']
        
        if not well_data['shutdown_time']:
            return False
        
        # Minimum shutdown duration
        min_shutdown_duration = timedelta(hours=1)
        if sensor_data.timestamp - well_data['shutdown_time'] < min_shutdown_duration:
            return False
        
        # Check that all safety parameters are within limits
        safe_conditions = [
            sensor_data.bottom_hole_pressure <= params.bhp_max,
            sensor_data.bottom_hole_pressure >= params.bhp_min,
            sensor_data.surface_pressure <= params.surface_pressure_max,
            params.temperature_min <= sensor_data.bottom_hole_temperature <= params.temperature_max,
        ]
        
        if sensor_data.seismic_magnitude is not None:
            safe_conditions.append(sensor_data.seismic_magnitude <= params.seismic_magnitude_threshold)
        
        if sensor_data.surface_co2_concentration is not None:
            safe_conditions.append(sensor_data.surface_co2_concentration <= params.surface_co2_concentration_threshold)
        
        return all(safe_conditions)
    
    def _is_cleanup_complete(self, well_name: str, sensor_data: WellSensorData) -> bool:
        """Determine if cleanup operations are complete"""
        well_data = self.wells[well_name]
        
        # Simple cleanup completion logic - could be enhanced with more sophisticated criteria
        cleanup_duration = timedelta(hours=6)  # Example cleanup duration
        last_transition = well_data['state_transitions'][-1] if well_data['state_transitions'] else None
        
        if last_transition and last_transition.to_state == WellControlState.CLEANUP:
            return sensor_data.timestamp - last_transition.timestamp >= cleanup_duration
        
        return False
    
    def _update_equipment_health(self, well_name: str, sensor_data: WellSensorData):
        """Update equipment health score based on sensor data"""
        well_data = self.wells[well_name]
        params = well_data['params']
        health_factors = []
        
        # Valve performance
        if sensor_data.valve_actuation_time is not None:
            valve_health = max(0, 1 - (sensor_data.valve_actuation_time /
                                     params.valve_actuation_time_threshold))
            health_factors.append(valve_health)
        
        # Corrosion monitoring
        if sensor_data.corrosion_rate is not None:
            corrosion_health = max(0, 1 - (sensor_data.corrosion_rate /
                                         params.corrosion_rate_threshold))
            health_factors.append(corrosion_health)
        
        # Pressure equipment health (simplified)
        pressure_health = 1.0 - min(1.0, sensor_data.surface_pressure /
                                  params.surface_pressure_max * 0.1)
        health_factors.append(pressure_health)
        
        if health_factors:
            well_data['equipment_health_score'] = np.mean(health_factors)
        
        # Trigger maintenance if health score drops below threshold
        # Use configurable maintenance threshold
        maintenance_threshold = getattr(well_data['params'], 'maintenance_health_threshold', 0.7)
        if (well_data['equipment_health_score'] < maintenance_threshold and
            well_data['current_state'] not in [WellControlState.SHUT_IN_MAINTENANCE, WellControlState.SHUT_IN_SAFETY]):
            logger.warning(f"Well {well_name}: Equipment health score low ({well_data['equipment_health_score']:.2f})")
            self._transition_state(well_name, WellControlState.SHUT_IN_MAINTENANCE,
                                 "Low equipment health score", sensor_data)
    
    def _transition_state(self, well_name: str, new_state: WellControlState, reason: str,
                         sensor_data: WellSensorData, safety_trigger: bool = False) -> WellControlState:
        """Execute state transition with logging and history tracking"""
        well_data = self.wells[well_name]
        
        if new_state == well_data['current_state']:
            return well_data['current_state']
        
        well_data['previous_state'] = well_data['current_state']
        well_data['current_state'] = new_state
        
        transition = WellControlStateTransition(
            timestamp=sensor_data.timestamp,
            from_state=well_data['previous_state'],
            to_state=new_state,
            reason=reason,
            sensor_data=sensor_data,
            triggered_by_safety=safety_trigger
        )
        
        well_data['state_transitions'].append(transition)
        
        logger.info(f"Well {well_name}: State transition {well_data['previous_state'].value} -> {new_state.value} - {reason}")
        
        return new_state
    
    def _record_control_history(self, well_name: str, sensor_data: WellSensorData):
        """Record control history for trend analysis and optimization"""
        well_data = self.wells[well_name]
        history_entry = {
            'timestamp': sensor_data.timestamp,
            'state': well_data['current_state'].value,
            'bhp': sensor_data.bottom_hole_pressure,
            'temperature': sensor_data.bottom_hole_temperature,
            'surface_pressure': sensor_data.surface_pressure,
            'injection_rate': sensor_data.injection_rate,
            'production_rate': sensor_data.production_rate,
            'equipment_health': well_data['equipment_health_score']
        }
        well_data['control_history'].append(history_entry)
    
    def get_control_setpoint(self, well_name: str, reservoir_state: Optional[Dict[str, Any]] = None) -> float:
        """
        Calculate optimal control setpoint based on current state and reservoir conditions
        """
        if well_name not in self.wells:
            raise ValueError(f"Well {well_name} not found in control system")
        
        well_data = self.wells[well_name]
        params = well_data['params']
        current_state = well_data['current_state']
        
        if current_state == WellControlState.INJECTOR_CO2:
            # For injectors, maintain BHP below fracture pressure with plume management
            injector_setpoint_factor = getattr(params, 'injection_setpoint_factor', 0.85)  # 85% of max BHP
            base_setpoint = params.bhp_max * injector_setpoint_factor
            
            if reservoir_state and params.plume_steering_enabled:
                # Adjust setpoint based on plume position and containment boundaries
                base_setpoint = self._adjust_for_plume_management(well_name, base_setpoint, reservoir_state)
            
            return base_setpoint
            
        elif current_state == WellControlState.PRODUCER:
            # For producers, maintain BHP above minimum to prevent coning/scaling
            producer_setpoint_factor = getattr(params, 'producer_setpoint_factor', 1.15)  # 15% above min BHP
            return params.bhp_min * producer_setpoint_factor
            
        elif current_state == WellControlState.OBSERVER:
            # For observers, maintain current pressure for monitoring
            if well_data['control_history']:
                last_entry = well_data['control_history'][-1]
                return last_entry.get('bhp', params.bhp_min)
            
        return params.control_setpoint
    
    def _adjust_for_plume_management(self, well_name: str, base_setpoint: float, reservoir_state: Dict[str, Any]) -> float:
        """Adjust injection setpoint for plume steering and containment"""
        well_data = self.wells[well_name]
        params = well_data['params']
        
        # Simplified plume management logic
        plume_distance = reservoir_state.get('plume_boundary_distance', float('inf'))
        # Use configurable containment safety margin
        containment_margin = getattr(params, 'containment_safety_margin_ft', 50.0)  # ft safety margin
        min_reduction_factor = getattr(params, 'min_injection_reduction_factor', 0.5)

        if plume_distance < params.containment_boundary_distance + containment_margin:
            # Reduce injection pressure when plume approaches boundary
            reduction_factor = max(min_reduction_factor, plume_distance /
                                 (params.containment_boundary_distance + containment_margin))
            return base_setpoint * reduction_factor
        
        return base_setpoint
    
    def optimize_multi_well_injection(self, reservoir_state: Dict[str, Any]) -> Dict[str, float]:
        """
        Optimize injection rates across multiple wells for capacity management
        Returns dictionary of well_name -> optimized_rate
        """
        
        well_group = [well for well in self.wells.values() if well['params'].capacity_optimization_enabled]

        if not well_group:
            return {}

        # Simple rate redistribution logic
        total_capacity = sum(w['params'].surface_rate_max for w in well_group)
        constrained_wells = []
        
        for well_data in well_group:
            if well_data['current_state'] == WellControlState.INJECTOR_CO2:
                # Check if well is approaching constraints
                if well_data['control_history']:
                    last_bhp = well_data['control_history'][-1].get('bhp', 0)
                    bhp_utilization = last_bhp / well_data['params'].bhp_max
                    
                    # Use configurable BHP utilization threshold
            bhp_utilization_threshold = getattr(well_data['params'], 'bhp_utilization_threshold', 0.8)  # 80% of max BHP
            if bhp_utilization > bhp_utilization_threshold:
                        constrained_wells.append((well_data, bhp_utilization))
        
        if not constrained_wells:
            return {}  # No optimization needed
        
        # Sort by constraint severity (highest utilization first)
        constrained_wells.sort(key=lambda x: x[1], reverse=True)
        
        optimization_results = {}
        total_reduction = 0.0
        
        for well_data, utilization in constrained_wells:
            # Calculate reduction needed (proportional to constraint severity)
            max_reduction_factor = getattr(well_data['params'], 'max_reduction_factor', 0.3)  # Max 30% reduction
            utilization_threshold = getattr(well_data['params'], 'utilization_threshold', 0.8)
            reduction_factor = min(max_reduction_factor, (utilization - utilization_threshold) * 2)
            current_rate = well_data['params'].control_setpoint
            new_rate = current_rate * (1 - reduction_factor)
            
            optimization_results[well_data['params'].well_name] = new_rate
            total_reduction += current_rate - new_rate
        
        # Distribute freed-up capacity to other injectors
        available_wells = [w for w in well_group
                          if w['current_state'] == WellControlState.INJECTOR_CO2
                          and w['params'].well_name not in optimization_results]
        
        if available_wells and total_reduction > 0:
            increment_per_well = total_reduction / len(available_wells)
            for well_data in available_wells:
                current_rate = well_data['params'].control_setpoint
                max_rate = well_data['params'].surface_rate_max
                new_rate = min(max_rate, current_rate + increment_per_well)
                optimization_results[well_data['params'].well_name] = new_rate
        
        return optimization_results
    
    def get_state_summary(self, well_name: str) -> Dict[str, Any]:
        """Get comprehensive summary of current well state and control parameters"""
        if well_name not in self.wells:
            raise ValueError(f"Well {well_name} not found in control system")
        
        well_data = self.wells[well_name]
        params = well_data['params']
        
        return {
            'well_name': params.well_name,
            'current_state': well_data['current_state'].value,
            'previous_state': well_data['previous_state'].value if well_data['previous_state'] else None,
            'control_mode': params.control_mode.value,
            'control_setpoint': params.control_setpoint,
            'equipment_health_score': well_data['equipment_health_score'],
            'total_injected_mass': well_data['total_injected_mass'],
            'transition_count': len(well_data['state_transitions']),
            'last_transition': well_data['state_transitions'][-1].reason if well_data['state_transitions'] else None,
            'constraints': {
                'bhp_max': params.bhp_max,
                'bhp_min': params.bhp_min,
                'surface_rate_max': params.surface_rate_max,
                'surface_pressure_max': params.surface_pressure_max
            }
        }
    
    def reset_for_maintenance(self, well_name: str):
        """Reset well state after maintenance completion"""
        if well_name not in self.wells:
            raise ValueError(f"Well {well_name} not found in control system")
            
        well_data = self.wells[well_name]
        well_data['equipment_health_score'] = 1.0
        self._transition_state(well_name, WellControlState.OBSERVER, "Maintenance completed",
                             WellSensorData(datetime.now(), 0, 0, 0, 0))