"""
Time-Stepping Cycle with User-Defined Intervals
Implements adaptive time-stepping for coupled CO₂-CCUS physics with user-defined resolutions
"""

import logging
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import datetime

logger = logging.getLogger(__name__)

class TimestepUnit(Enum):
    """Supported time-stepping units"""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    YEARLY = "yearly"

@dataclass
class TimestepConfig:
    """Configuration for time-stepping behavior"""
    unit: TimestepUnit
    base_dt_days: float  # Base timestep in days
    max_dt_days: float  # Maximum allowed timestep
    min_dt_days: float  # Minimum allowed timestep
    adaptive_stepping: bool = True
    max_nonlinear_iterations: int = 20
    convergence_tolerance: float = 1e-4
    
    def get_timestep_seconds(self) -> float:
        """Convert timestep to seconds"""
        return self.base_dt_days * 24 * 3600

@dataclass
class SimulationSchedule:
    """Defines the simulation schedule with user-defined timesteps"""
    start_date: datetime.datetime
    end_date: datetime.datetime
    timestep_config: TimestepConfig
    injection_schedule: Dict[datetime.datetime, Dict[str, float]]  # Injection rates by date
    output_frequency: TimestepUnit = TimestepUnit.MONTHLY
    
    def get_total_days(self) -> float:
        """Calculate total simulation days"""
        delta = self.end_date - self.start_date
        return delta.days
    
    def generate_timesteps(self) -> List[float]:
        """Generate timestep sequence in days"""
        total_days = self.get_total_days()
        base_dt = self.timestep_config.base_dt_days
        
        timesteps = []
        current_day = 0.0
        
        while current_day < total_days:
            dt = min(base_dt, total_days - current_day)
            timesteps.append(dt)
            current_day += dt
        
        return timesteps

@dataclass
class TimestepResult:
    """Results from a single timestep"""
    timestep_index: int
    simulation_time_days: float
    calendar_date: datetime.datetime
    dt_days: float
    state_snapshot: Dict[str, Any]
    convergence_info: Dict[str, Any]
    performance_metrics: Dict[str, float]
    
    def validate(self) -> bool:
        """Validate timestep results"""
        required_fields = ['pressure', 'saturations', 'porosity', 'permeability']
        for field in required_fields:
            if field not in self.state_snapshot:
                logger.warning(f"Missing field in state snapshot: {field}")
                return False
        return True

class AdaptiveTimeStepper:
    """
    Adaptive time-stepping controller for coupled physics
    Adjusts timestep size based on convergence behavior and physical changes
    """

    # Mass balance tolerance constants
    MASS_BALANCE_TOLERANCE = 1e-6  # 0.0001% - strict tolerance for mass conservation

    def __init__(self, config: TimestepConfig):
        self.config = config
        self.current_dt = config.base_dt_days
        self.dt_history: List[float] = []
        self.convergence_history: List[bool] = []

        # Adaptive stepping parameters
        self.dt_increase_factor = 1.5
        self.dt_decrease_factor = 0.5
        self.max_dt_increase = 2.0
        self.min_iterations_for_increase = 3
        
    def calculate_initial_timestep(self, initial_conditions: Dict[str, Any]) -> float:
        """Calculate initial timestep based on physical conditions"""
        # Use base timestep as starting point
        dt = self.config.base_dt_days
        
        # Adjust based on injection rates if available
        if 'injection_rates' in initial_conditions:
            max_injection = max(initial_conditions['injection_rates'].values()) if initial_conditions['injection_rates'] else 0.0
            if max_injection > 0:
                # Reduce timestep for high injection rates
                injection_factor = min(1.0, 1000.0 / max_injection)  # Adjust based on experience
                dt *= injection_factor
        
        # Clamp to allowed range
        dt = np.clip(dt, self.config.min_dt_days, self.config.max_dt_days)
        
        logger.info(f"Initial timestep: {dt:.2f} days")
        return dt
    
    def adjust_timestep(self, convergence: bool, iterations: int,
                       physical_changes: Dict[str, float],
                       mass_balance_error: float = 0.0) -> float:
        """
        Adjust timestep based on convergence behavior, physical changes, and mass balance.

        Args:
            convergence: Whether the timestep converged
            iterations: Number of iterations taken
            physical_changes: Dict of physical variable changes
            mass_balance_error: Relative mass balance error (fraction)

        Returns:
            Adjusted timestep size (days)
        """
        if not self.config.adaptive_stepping:
            return self.current_dt

        old_dt = self.current_dt

        # Check mass balance error FIRST - this is critical for conservation
        if mass_balance_error > self.MASS_BALANCE_TOLERANCE:
            logger.warning(
                f"Mass balance error {mass_balance_error*100:.6f}% exceeds tolerance "
                f"({self.MASS_BALANCE_TOLERANCE*100:.6f}%). Reducing timestep."
            )
            # Reduce timestep significantly for mass balance issues
            new_dt = max(old_dt * self.dt_decrease_factor * 0.5, self.config.min_dt_days)
            self.current_dt = new_dt
            self.dt_history.append(new_dt)
            return new_dt

        if convergence:
            # Successful convergence - consider increasing timestep
            if iterations < self.min_iterations_for_increase:
                # Fast convergence - increase timestep
                new_dt = min(old_dt * self.dt_increase_factor,
                           self.config.max_dt_days)

                # Check physical changes - don't increase if changes are large
                max_change = max(physical_changes.values()) if physical_changes else 0.0
                if max_change > 0.1:  # More than 10% change in any variable
                    new_dt = old_dt  # Keep current timestep

            else:
                # Normal convergence - maintain current timestep
                new_dt = old_dt
        else:
            # Convergence failed - decrease timestep
            new_dt = max(old_dt * self.dt_decrease_factor,
                        self.config.min_dt_days)
            logger.warning(f"Reducing timestep due to convergence failure: {old_dt:.2f} -> {new_dt:.2f} days")

        # Update history
        self.dt_history.append(new_dt)
        self.convergence_history.append(convergence)

        self.current_dt = new_dt
        return new_dt
    
    def calculate_physical_changes(self, current_state: Dict[str, Any], 
                                 previous_state: Dict[str, Any]) -> Dict[str, float]:
        """Calculate maximum changes in key physical variables"""
        changes = {}
        
        # Pressure change
        if 'pressure' in current_state and 'pressure' in previous_state:
            pressure_change = np.max(np.abs(current_state['pressure'] - previous_state['pressure']))
            changes['pressure'] = pressure_change
        
        # Saturation changes
        if 'saturations' in current_state and 'saturations' in previous_state:
            saturation_change = np.max(np.abs(current_state['saturations'] - previous_state['saturations']))
            changes['saturation'] = saturation_change
        
        # Porosity change
        if 'porosity' in current_state and 'porosity' in previous_state:
            porosity_change = np.max(np.abs(current_state['porosity'] - previous_state['porosity']))
            changes['porosity'] = porosity_change
        
        return changes

class CoupledPhysicsTimeStepper:
    """
    Main time-stepping controller for coupled CO₂-CCUS physics
    Orchestrates the solution sequence across all physics modules
    """
    
    def __init__(self, physics_engine: Any, schedule: SimulationSchedule):
        self.physics_engine = physics_engine
        self.schedule = schedule
        self.adaptive_stepper = AdaptiveTimeStepper(schedule.timestep_config)
        
        # Simulation state
        self.current_time = 0.0
        self.current_date = schedule.start_date
        self.timestep_results: List[TimestepResult] = []
        self.state_history: List[Dict[str, Any]] = []
        
        # Performance tracking
        self.iteration_counts: List[int] = []
        self.convergence_flags: List[bool] = []
        
    def initialize_simulation(self, initial_conditions: Dict[str, Any]):
        """Initialize simulation with initial conditions"""
        logger.info("Initializing coupled physics simulation")
        
        # Set initial timestep
        initial_dt = self.adaptive_stepper.calculate_initial_timestep(initial_conditions)
        
        # Store initial state
        self.state_history.append(initial_conditions)
        
        logger.info(f"Simulation initialized: {self.schedule.start_date} to {self.schedule.end_date}")
        logger.info(f"Total days: {self.schedule.get_total_days():.1f}")
        logger.info(f"Initial timestep: {initial_dt:.2f} days")
    
    def run_timestep(self) -> TimestepResult:
        """Run one coupled physics timestep"""
        timestep_index = len(self.timestep_results)
        dt = self.adaptive_stepper.current_dt
        
        logger.info(f"Timestep {timestep_index + 1}: t={self.current_time:.1f} days, dt={dt:.2f} days")
        
        # Get injection rates for current time
        injection_rates = self._get_current_injection_rates()
        
        # Run coupled physics for this timestep
        start_time = datetime.datetime.now()
        
        try:
            # Solve coupled physics
            convergence, iterations, new_state, dt = self._solve_coupled_physics(dt, injection_rates, timestep_index)
            
            # Calculate physical changes
            previous_state = self.state_history[-1] if self.state_history else new_state
            physical_changes = self.adaptive_stepper.calculate_physical_changes(new_state, previous_state)
            
            # Adjust timestep for next step
            next_dt = self.adaptive_stepper.adjust_timestep(convergence, iterations, physical_changes)
            
            # Calculate performance metrics
            end_time = datetime.datetime.now()
            solve_time = (end_time - start_time).total_seconds()
            performance_metrics = {
                'solve_time_seconds': solve_time,
                'iterations': iterations,
                'convergence': convergence,
                'max_pressure_change': physical_changes.get('pressure', 0.0),
                'max_saturation_change': physical_changes.get('saturation', 0.0),
                'next_timestep_days': next_dt
            }
            
            # Create timestep result
            result = TimestepResult(
                timestep_index=timestep_index,
                simulation_time_days=self.current_time,
                calendar_date=self.current_date,
                dt_days=dt,
                state_snapshot=new_state,
                convergence_info={
                    'converged': convergence,
                    'iterations': iterations,
                    'residual_norm': physical_changes.get('pressure', 0.0)  # Simplified
                },
                performance_metrics=performance_metrics
            )
            
            # Update simulation state
            if convergence:
                self.current_time += dt
                self.current_date += datetime.timedelta(days=dt)
                self.state_history.append(new_state)
            
            self.timestep_results.append(result)
            self.iteration_counts.append(iterations)
            self.convergence_flags.append(convergence)
            
            logger.info(f"Timestep {timestep_index + 1} completed: "
                       f"converged={convergence}, iterations={iterations}, "
                       f"time={solve_time:.2f}s")
            
            return result
            
        except Exception as e:
            logger.error(f"Timestep {timestep_index + 1} failed: {e}")
            # Create error result
            return TimestepResult(
                timestep_index=timestep_index,
                simulation_time_days=self.current_time,
                calendar_date=self.current_date,
                dt_days=dt,
                state_snapshot={},
                convergence_info={'converged': False, 'error': str(e)},
                performance_metrics={'solve_time_seconds': 0, 'iterations': 0, 'convergence': False}
            )
    
    def _solve_coupled_physics(self, dt: float, injection_rates: Dict[str, float], timestep_index: int) -> Tuple[bool, int, Dict[str, Any], float]:
        """
        Solve coupled physics for one timestep
        Returns: (convergence_flag, iterations, new_state, dt)
        """
        try:
            new_state_obj, dt = self.physics_engine.run_timestep(dt, injection_rates, timestep_index=timestep_index)
            return True, 1, new_state_obj.__dict__, dt
        except Exception as e:
            logger.error(f"Coupled iteration failed: {e}")
            return False, 1, {}, dt
    
    def _check_convergence(self, new_state: Dict[str, Any], old_state: Dict[str, Any], 
                          tolerance: float) -> bool:
        """Check convergence between successive iterations"""
        if not old_state:
            return False
        
        # Check pressure convergence
        if 'pressure' in new_state and 'pressure' in old_state:
            pressure_residual = np.max(np.abs(new_state['pressure'] - old_state['pressure']))
            if pressure_residual > tolerance:
                return False
        
        # Check saturation convergence
        if 'saturations' in new_state and 'saturations' in old_state:
            saturation_residual = np.max(np.abs(new_state['saturations'] - old_state['saturations']))
            if saturation_residual > tolerance:
                return False
        
        return True
    
    def _get_current_injection_rates(self) -> Dict[str, float]:
        """Get injection rates for current simulation time"""
        # Find the closest injection schedule entry
        current_date = self.current_date
        
        # Look for exact match or most recent schedule
        if current_date in self.schedule.injection_schedule:
            return self.schedule.injection_schedule[current_date]
        
        # Find the most recent schedule entry before current date
        previous_schedules = {date: rates for date, rates in self.schedule.injection_schedule.items() 
                            if date <= current_date}
        
        if previous_schedules:
            # Get the most recent schedule
            latest_date = max(previous_schedules.keys())
            return previous_schedules[latest_date]
        
        return {}  # No injection
    
    def run_full_simulation(self) -> List[TimestepResult]:
        """Run the full simulation from start to end"""
        logger.info("Starting full coupled physics simulation")
        
        total_days = self.schedule.get_total_days()
        timestep_count = 0
        max_timesteps = int(total_days / self.schedule.timestep_config.min_dt_days) * 10  # Safety limit
        
        while self.current_time < total_days and timestep_count < max_timesteps:
            result = self.run_timestep()
            
            if not result.convergence_info.get('converged', False):
                logger.warning(f"Timestep {timestep_count + 1} did not converge")
                # Continue with next timestep anyway for now
            
            timestep_count += 1
            
            # Progress reporting
            if timestep_count % 10 == 0:
                progress = (self.current_time / total_days) * 100
                logger.info(f"Simulation progress: {progress:.1f}% "
                           f"({self.current_time:.1f}/{total_days:.1f} days)")
        
        logger.info(f"Simulation completed: {timestep_count} timesteps, "
                   f"final time: {self.current_time:.1f} days")
        
        return self.timestep_results
    
    def get_simulation_summary(self) -> Dict[str, Any]:
        """Get summary of simulation performance and results"""
        if not self.timestep_results:
            return {}
        
        total_timesteps = len(self.timestep_results)
        converged_timesteps = sum(self.convergence_flags)
        convergence_rate = converged_timesteps / total_timesteps if total_timesteps > 0 else 0.0
        
        avg_iterations = np.mean(self.iteration_counts) if self.iteration_counts else 0.0
        avg_timestep = np.mean([r.dt_days for r in self.timestep_results]) if self.timestep_results else 0.0
        
        total_solve_time = sum(r.performance_metrics.get('solve_time_seconds', 0) 
                              for r in self.timestep_results)
        
        return {
            'total_timesteps': total_timesteps,
            'converged_timesteps': converged_timesteps,
            'convergence_rate': convergence_rate,
            'average_iterations': avg_iterations,
            'average_timestep_days': avg_timestep,
            'total_solve_time_seconds': total_solve_time,
            'final_simulation_time_days': self.current_time,
            'final_calendar_date': self.current_date.isoformat() if self.current_date else None
        }

# Utility functions for common time-stepping scenarios
def create_daily_timestepping(start_date: datetime.datetime, end_date: datetime.datetime) -> SimulationSchedule:
    """Create daily time-stepping configuration"""
    config = TimestepConfig(
        unit=TimestepUnit.DAILY,
        base_dt_days=1.0,
        max_dt_days=7.0,  # Allow up to weekly for adaptive stepping
        min_dt_days=0.1,
        adaptive_stepping=True
    )
    
    return SimulationSchedule(
        start_date=start_date,
        end_date=end_date,
        timestep_config=config,
        injection_schedule={},
        output_frequency=TimestepUnit.DAILY
    )

def create_weekly_timestepping(start_date: datetime.datetime, end_date: datetime.datetime) -> SimulationSchedule:
    """Create weekly time-stepping configuration"""
    config = TimestepConfig(
        unit=TimestepUnit.WEEKLY,
        base_dt_days=7.0,
        max_dt_days=30.0,  # Allow up to monthly
        min_dt_days=1.0,
        adaptive_stepping=True
    )
    
    return SimulationSchedule(
        start_date=start_date,
        end_date=end_date,
        timestep_config=config,
        injection_schedule={},
        output_frequency=TimestepUnit.WEEKLY
    )

def create_monthly_timestepping(start_date: datetime.datetime, end_date: datetime.datetime) -> SimulationSchedule:
    """Create monthly time-stepping configuration"""
    config = TimestepConfig(
        unit=TimestepUnit.MONTHLY,
        base_dt_days=30.0,
        max_dt_days=90.0,  # Allow up to quarterly
        min_dt_days=7.0,
        adaptive_stepping=True
    )
    
    return SimulationSchedule(
        start_date=start_date,
        end_date=end_date,
        timestep_config=config,
        injection_schedule={},
        output_frequency=TimestepUnit.MONTHLY
    )

def create_yearly_timestepping(start_date: datetime.datetime, end_date: datetime.datetime) -> SimulationSchedule:
    """Create yearly time-stepping configuration"""
    config = TimestepConfig(
        unit=TimestepUnit.YEARLY,
        base_dt_days=365.0,
        max_dt_days=730.0,  # Allow up to 2 years
        min_dt_days=90.0,
        adaptive_stepping=True
    )
    
    return SimulationSchedule(
        start_date=start_date,
        end_date=end_date,
        timestep_config=config,
        injection_schedule={},
        output_frequency=TimestepUnit.YEARLY
    )