from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import logging
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import random
from scipy.optimize import minimize

@dataclass
class WellData:
    """Container for well log data"""
    name: str
    depths: np.ndarray
    properties: Dict[str, np.ndarray]
    units: Dict[str, str]
    
    def validate(self) -> bool:
        """Check data consistency"""
        return len(self.depths) == len(next(iter(self.properties.values())))

@dataclass
class ReservoirData:
    """Container for reservoir simulation data"""
    grid: Dict[str, np.ndarray]
    pvt_tables: Dict[str, np.ndarray]
    regions: Optional[Dict[str, np.ndarray]] = None

@dataclass
class PVTProperties:
    """PVT properties for oil and gas"""
    oil_fvf: np.ndarray
    oil_viscosity: np.ndarray
    gas_fvf: np.ndarray
    gas_viscosity: np.ndarray
    rs: np.ndarray  # Solution GOR
    pvt_type: str  # 'black_oil' or 'compositional'

@dataclass
class EORParameters:
    """CO2 EOR operational parameters"""
    injection_rate: float
    WAG_ratio: Optional[float] = None
    injection_scheme: str = 'continuous'  # or 'wag'
    target_pressure: float = 0.0
    max_pressure: float = 5000.0  # psi
    min_injection_rate: float = 1000.0  # bbl/day

@dataclass
class GeneticAlgorithmParams:
    """Parameters for genetic algorithm optimization"""
    population_size: int = 50
    generations: int = 100
    crossover_rate: float = 0.8
    mutation_rate: float = 0.1
    selection_pressure: float = 1.5
    elite_count: int = 2

def recovery_factor(pressure: float, rate: float, porosity: float, mmp: float) -> float:
    """Estimate recovery factor based on injection parameters"""
    miscibility = min(1.0, max(0.0, (pressure - mmp) / mmp))
    sweep_efficiency = 0.7 * (rate ** 0.2) * (porosity ** 0.5)
    return min(0.7, miscibility * sweep_efficiency)

class OptimizationEngine:
    """Core optimization engine for CO2 EOR processes with parallel computing support"""
    
    def __init__(self, reservoir: ReservoirData, pvt: PVTProperties, eor_params: EORParameters):
        self.reservoir = reservoir
        self.pvt = pvt
        self.eor_params = eor_params
        self._results = None
        self._mmp = None
        self._mmp_params = None
        
    def calculate_mmp(self, method: str = 'auto') -> float:
        """Calculate Minimum Miscibility Pressure (MMP)"""
        from .evaluation.mmp import calculate_mmp, MMPParameters
        
        result = calculate_mmp(self.pvt, method)
        logging.debug(f"MMP Calculation Result: {result} (Type: {type(result)})")
        
        if isinstance(result, MMPParameters):
            self._mmp_params = result
            self._mmp_value = float(calculate_mmp(result, method))
        else:
            self._mmp_value = float(result)
            self._mmp_params = MMPParameters(
                temperature=180,
                oil_gravity=35,
                pvt_data=self.pvt
            )
        
        self._mmp = self._mmp_value  # Ensure both variables are set
            
        return self._mmp_value
        
    def optimize_recovery(self, max_iter: int = 100, tol: float = 1e-4,
                        learning_rate: float = 50.0) -> Dict[str, float]:
        """Optimize CO2 injection parameters using gradient descent"""
        if not hasattr(self, '_mmp_value'):
            self.calculate_mmp()
            
        avg_poro = np.mean(self.reservoir.grid['PORO'])
        pressure = max(float(self._mmp_value), self.eor_params.target_pressure)
        rate = self.eor_params.injection_rate
        prev_recovery = 0.0
        converged = False
        
        for i in range(max_iter):
            recovery = recovery_factor(pressure, rate, avg_poro, self._mmp_value)
            
            if abs(recovery - prev_recovery) < tol and i > 5:
                converged = True
                break
                
            delta_p = 10.0
            recovery_plus = recovery_factor(pressure + delta_p, rate, avg_poro, self._mmp_value)
            gradient = (recovery_plus - recovery) / delta_p
            
            new_pressure = pressure + learning_rate * gradient
            pressure = max(self._mmp_value, min(new_pressure, 1.5 * self._mmp_value))
            prev_recovery = recovery
            
        self._results = {
            'optimized_params': {
                'injection_rate': rate,
                'target_pressure': pressure
            },
            'mmp': self._mmp_value,
            'iterations': i,
            'final_recovery': recovery,
            'converged': converged,
            'avg_porosity': avg_poro
        }
        return self._results
        
    def optimize_genetic_algorithm(self, ga_params: GeneticAlgorithmParams = None) -> Dict[str, Any]:
        """Optimize using genetic algorithm with parallel evaluation"""
        if not hasattr(self, '_mmp_value'):
            self.calculate_mmp()
            
        params = ga_params or GeneticAlgorithmParams()
        avg_poro = np.mean(self.reservoir.grid['PORO'])
        
        # Initialize population
        population = self._initialize_population(params.population_size)
        
        best_solution = None
        best_fitness = -np.inf
        
        for generation in range(params.generations):
            # Parallel fitness evaluation
            with ProcessPoolExecutor() as executor:
                fitnesses = list(executor.map(
                    partial(self._evaluate_individual, avg_poro=avg_poro, mmp=self._mmp_value),
                    population
                ))
            
            # Selection
            selected = self._tournament_selection(population, fitnesses, 
                                                params.selection_pressure, 
                                                params.elite_count)
            
            # Crossover
            offspring = self._crossover(selected, params.crossover_rate)
            
            # Mutation
            population = self._mutate(offspring, params.mutation_rate)
            
            # Update best solution
            current_best_idx = np.argmax(fitnesses)
            if fitnesses[current_best_idx] > best_fitness:
                best_fitness = fitnesses[current_best_idx]
                best_solution = population[current_best_idx]
                
        # Convert best solution to operational parameters
        pressure, rate = best_solution
        self._results = {
            'optimized_params': {
                'injection_rate': rate,
                'target_pressure': pressure
            },
            'mmp': self._mmp_value,
            'method': 'genetic_algorithm',
            'generations': params.generations,
            'population_size': params.population_size,
            'final_recovery': best_fitness,
            'avg_porosity': avg_poro,
            'converged': True,
            'iterations': params.generations
        }
        return self._results
        
    def _initialize_population(self, size: int) -> List[Tuple[float, float]]:
        """Initialize population within operational constraints"""
        population = []
        min_pressure = max(self._mmp_value, self.eor_params.target_pressure)
        
        for _ in range(size):
            pressure = random.uniform(min_pressure, self.eor_params.max_pressure)
            rate = random.uniform(self.eor_params.min_injection_rate, 
                                self.eor_params.injection_rate * 1.5)
            population.append((pressure, rate))
            
        return population
        
    def _evaluate_individual(self, individual: Tuple[float, float], 
                           avg_poro: float, mmp: float) -> float:
        """Evaluate fitness of an individual solution"""
        pressure, rate = individual
        return recovery_factor(pressure, rate, avg_poro, mmp)
        
    def _tournament_selection(self, population: List[Tuple[float, float]], 
                            fitnesses: List[float], pressure: float, 
                            elite_count: int) -> List[Tuple[float, float]]:
        """Select individuals using tournament selection"""
        selected = []
        population_size = len(population)
        
        # Keep elites
        elite_indices = np.argsort(fitnesses)[-elite_count:]
        selected.extend([population[i] for i in elite_indices])
        
        # Tournament selection for rest
        for _ in range(population_size - elite_count):
            candidates = random.sample(range(population_size), 2)
            winner = max(candidates, key=lambda x: fitnesses[x])
            selected.append(population[winner])
            
        return selected
        
    def _crossover(self, parents: List[Tuple[float, float]], rate: float) -> List[Tuple[float, float]]:
        """Perform crossover between parents"""
        offspring = []
        for i in range(0, len(parents), 2):
            if i + 1 >= len(parents):
                offspring.append(parents[i])
                continue
                
            parent1, parent2 = parents[i], parents[i+1]
            
            if random.random() < rate:
                # Blend crossover
                alpha = random.random()
                child1 = (
                    alpha * parent1[0] + (1 - alpha) * parent2[0],
                    alpha * parent1[1] + (1 - alpha) * parent2[1]
                )
                child2 = (
                    (1 - alpha) * parent1[0] + alpha * parent2[0],
                    (1 - alpha) * parent1[1] + alpha * parent2[1]
                )
                offspring.extend([child1, child2])
            else:
                offspring.extend([parent1, parent2])
                
        return offspring
        
    def _mutate(self, population: List[Tuple[float, float]], rate: float) -> List[Tuple[float, float]]:
        """Apply mutation to population"""
        mutated = []
        min_pressure = max(self._mmp_value, self.eor_params.target_pressure)
        
        for individual in population:
            if random.random() < rate:
                # Gaussian mutation
                pressure = np.clip(
                    individual[0] + random.gauss(0, 100),
                    min_pressure, self.eor_params.max_pressure
                )
                rate_val = np.clip(
                    individual[1] + random.gauss(0, 200),
                    self.eor_params.min_injection_rate, 
                    self.eor_params.injection_rate * 1.5
                )
                mutated.append((pressure, rate_val))
            else:
                mutated.append(individual)
                
        return mutated
        
    def optimize_wag(self, cycles: int = 5, max_iter: int = 20) -> Dict[str, float]:
        """Optimize Water-Alternating-Gas (WAG) parameters"""
        if not hasattr(self, '_mmp_value'):
            self.calculate_mmp()
            
        avg_poro = np.mean(self.reservoir.grid['PORO'])
        
        min_cycle = 15
        max_cycle = 90
        min_water = 0.2
        max_water = 0.8
        
        best_params = {
            'cycle_length': 30,
            'water_fraction': 0.5,
            'injection_pressure': max(float(self._mmp_value), self.eor_params.target_pressure),
            'recovery': 0.0
        }
        
        for cycle in range(cycles):
            for water_frac in np.linspace(min_water, max_water, num=5):
                for cycle_len in np.linspace(min_cycle, max_cycle, num=5):
                    effective_rate = self.eor_params.injection_rate * (1 - water_frac)
                    recovery = recovery_factor(
                        best_params['injection_pressure'],
                        effective_rate,
                        avg_poro,
                        self._mmp_value
                    )
                    
                    if recovery > best_params['recovery']:
                        best_params.update({
                            'cycle_length': cycle_len,
                            'water_fraction': water_frac,
                            'recovery': recovery
                        })
            
            min_water = max(0.1, best_params['water_fraction'] - 0.2)
            max_water = min(0.9, best_params['water_fraction'] + 0.2)
            min_cycle = max(10, best_params['cycle_length'] - 20)
            max_cycle = min(120, best_params['cycle_length'] + 20)
        
        pressure = self._optimize_pressure_for_wag(
            best_params['water_fraction'],
            best_params['cycle_length'],
            max_iter
        )
        
        self._results = {
            'wag_params': {
                'optimal_cycle_length': best_params['cycle_length'],
                'optimal_water_fraction': best_params['water_fraction'],
                'injection_pressure': pressure,
                'sensitivity': {
                    'water_fraction': (min_water, max_water),
                    'cycle_length': (min_cycle, max_cycle)
                }
            },
            'mmp': self._mmp_value,
            'estimated_recovery': best_params['recovery'],  # At root level
            'avg_porosity': avg_poro
        }
        return self._results
        
    def _optimize_pressure_for_wag(self, water_frac: float, cycle_len: int,
                                 max_iter: int = 20) -> float:
        """Optimize injection pressure for given WAG parameters"""
        avg_poro = np.mean(self.reservoir.grid['PORO'])
        effective_rate = self.eor_params.injection_rate * (1 - water_frac)
        
        pressure = max(float(self._mmp_value), self.eor_params.target_pressure)
        best_pressure = pressure
        best_recovery = 0.0
        
        for _ in range(max_iter):
            recovery = recovery_factor(
                pressure,
                effective_rate,
                avg_poro,
                self._mmp_value
            )
            
            if recovery > best_recovery:
                best_recovery = recovery
                best_pressure = pressure
                
            delta_p = 50
            recovery_plus = recovery_factor(
                pressure + delta_p,
                effective_rate,
                avg_poro,
                self._mmp_value
            )
            
            gradient = (recovery_plus - recovery) / delta_p
            pressure += 100 * gradient
            
            pressure = max(
                self._mmp_value * 1.05,
                min(pressure, self._mmp_value * 1.5)
            )
            
        return best_pressure
        
    def check_mmp_constraint(self, pressure: float) -> bool:
        """Check if pressure meets Minimum Miscibility Pressure (MMP) constraint"""
        if not hasattr(self, '_mmp_value'):
            self.calculate_mmp()
        return pressure >= float(self._mmp_value)
        
    @property
    def results(self) -> Optional[Dict]:
        """Get optimization results"""
        return self._results
        
    @property
    def mmp(self) -> Optional[float]:
        """Get calculated MMP value"""
        return self._mmp_value if hasattr(self, '_mmp_value') else None