"""
Parameter Tuning Module

This module provides automated parameter optimization for blockchain network parameters.
"""

import logging

logger = logging.getLogger(__name__)
import numpy as np
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Callable
from enum import Enum
import json
import random

from ..errors import ClientError
from dubchain.logging import get_logger

logger = get_logger(__name__)

class OptimizationAlgorithm(Enum):
    """Optimization algorithms available."""
    GRID_SEARCH = "grid_search"
    RANDOM_SEARCH = "random_search"
    BAYESIAN_OPTIMIZATION = "bayesian_optimization"
    GENETIC_ALGORITHM = "genetic_algorithm"
    PARTICLE_SWARM = "particle_swarm"

@dataclass
class ParameterRange:
    """Parameter range definition."""
    name: str
    min_value: float
    max_value: float
    step: Optional[float] = None
    param_type: str = "float"  # float, int, bool, categorical

@dataclass
class OptimizationResult:
    """Result of parameter optimization."""
    best_parameters: Dict[str, Any]
    best_score: float
    optimization_history: List[Dict[str, Any]]
    algorithm_used: OptimizationAlgorithm
    total_evaluations: int
    optimization_time: float
    convergence_info: Dict[str, Any] = field(default_factory=dict)

@dataclass
class OptimizationConfig:
    """Configuration for parameter optimization."""
    algorithm: OptimizationAlgorithm = OptimizationAlgorithm.BAYESIAN_OPTIMIZATION
    max_evaluations: int = 100
    convergence_threshold: float = 0.001
    patience: int = 10
    random_seed: Optional[int] = None
    parallel_evaluations: int = 1

class GridSearchOptimizer:
    """Grid search parameter optimizer."""
    
    def __init__(self, config: OptimizationConfig):
        """Initialize grid search optimizer."""
        self.config = config
        self.evaluation_count = 0
        self.best_score = float('-inf')
        self.best_parameters = {}
        self.history = []
        
        logger.info("Initialized Grid Search optimizer")
    
    def optimize(self, 
                 parameter_ranges: List[ParameterRange], 
                 objective_function: Callable[[Dict[str, Any]], float]) -> OptimizationResult:
        """Perform grid search optimization."""
        try:
            start_time = time.time()
            
            # Generate parameter combinations
            parameter_combinations = self._generate_parameter_combinations(parameter_ranges)
            
            logger.info(f"Grid search: evaluating {len(parameter_combinations)} parameter combinations")
            
            # Evaluate each combination
            for i, params in enumerate(parameter_combinations):
                if self.evaluation_count >= self.config.max_evaluations:
                    break
                
                score = objective_function(params)
                self.evaluation_count += 1
                
                # Track best result
                if score > self.best_score:
                    self.best_score = score
                    self.best_parameters = params.copy()
                
                # Record history
                self.history.append({
                    'evaluation': self.evaluation_count,
                    'parameters': params.copy(),
                    'score': score,
                    'timestamp': time.time()
                })
                
                if i % 10 == 0:
                    logger.info(f"Grid search progress: {i}/{len(parameter_combinations)}")
            
            optimization_time = time.time() - start_time
            
            return OptimizationResult(
                best_parameters=self.best_parameters,
                best_score=self.best_score,
                optimization_history=self.history,
                algorithm_used=OptimizationAlgorithm.GRID_SEARCH,
                total_evaluations=self.evaluation_count,
                optimization_time=optimization_time,
                convergence_info={"method": "grid_search", "total_combinations": len(parameter_combinations)}
            )
            
        except Exception as e:
            logger.error(f"Error in grid search optimization: {e}")
            raise ClientError(f"Grid search optimization failed: {str(e)}")
    
    def _generate_parameter_combinations(self, parameter_ranges: List[ParameterRange]) -> List[Dict[str, Any]]:
        """Generate all parameter combinations for grid search."""
        try:
            combinations = []
            
            # Generate parameter values for each range
            param_values = {}
            for param_range in parameter_ranges:
                if param_range.param_type == "float":
                    if param_range.step:
                        values = np.arange(param_range.min_value, param_range.max_value + param_range.step, param_range.step)
                    else:
                        # Default to 10 values
                        values = np.linspace(param_range.min_value, param_range.max_value, 10)
                elif param_range.param_type == "int":
                    if param_range.step:
                        values = list(range(int(param_range.min_value), int(param_range.max_value) + 1, int(param_range.step)))
                    else:
                        values = list(range(int(param_range.min_value), int(param_range.max_value) + 1))
                elif param_range.param_type == "bool":
                    values = [True, False]
                else:  # categorical
                    values = [param_range.min_value, param_range.max_value]
                
                param_values[param_range.name] = values
            
            # Generate all combinations
            from itertools import product
            
            for combination in product(*param_values.values()):
                param_dict = dict(zip(param_values.keys(), combination))
                combinations.append(param_dict)
            
            return combinations
            
        except Exception as e:
            logger.error(f"Error generating parameter combinations: {e}")
            return []

class RandomSearchOptimizer:
    """Random search parameter optimizer."""
    
    def __init__(self, config: OptimizationConfig):
        """Initialize random search optimizer."""
        self.config = config
        self.evaluation_count = 0
        self.best_score = float('-inf')
        self.best_parameters = {}
        self.history = []
        
        if config.random_seed:
            random.seed(config.random_seed)
            np.random.seed(config.random_seed)
        
        logger.info("Initialized Random Search optimizer")
    
    def optimize(self, 
                 parameter_ranges: List[ParameterRange], 
                 objective_function: Callable[[Dict[str, Any]], float]) -> OptimizationResult:
        """Perform random search optimization."""
        try:
            start_time = time.time()
            
            logger.info(f"Random search: performing {self.config.max_evaluations} evaluations")
            
            # Perform random evaluations
            for i in range(self.config.max_evaluations):
                # Generate random parameters
                params = self._generate_random_parameters(parameter_ranges)
                
                # Evaluate
                score = objective_function(params)
                self.evaluation_count += 1
                
                # Track best result
                if score > self.best_score:
                    self.best_score = score
                    self.best_parameters = params.copy()
                
                # Record history
                self.history.append({
                    'evaluation': self.evaluation_count,
                    'parameters': params.copy(),
                    'score': score,
                    'timestamp': time.time()
                })
                
                if i % 10 == 0:
                    logger.info(f"Random search progress: {i}/{self.config.max_evaluations}")
            
            optimization_time = time.time() - start_time
            
            return OptimizationResult(
                best_parameters=self.best_parameters,
                best_score=self.best_score,
                optimization_history=self.history,
                algorithm_used=OptimizationAlgorithm.RANDOM_SEARCH,
                total_evaluations=self.evaluation_count,
                optimization_time=optimization_time,
                convergence_info={"method": "random_search"}
            )
            
        except Exception as e:
            logger.error(f"Error in random search optimization: {e}")
            raise ClientError(f"Random search optimization failed: {str(e)}")
    
    def _generate_random_parameters(self, parameter_ranges: List[ParameterRange]) -> Dict[str, Any]:
        """Generate random parameters within ranges."""
        params = {}
        
        for param_range in parameter_ranges:
            if param_range.param_type == "float":
                value = random.uniform(param_range.min_value, param_range.max_value)
            elif param_range.param_type == "int":
                value = random.randint(int(param_range.min_value), int(param_range.max_value))
            elif param_range.param_type == "bool":
                value = random.choice([True, False])
            else:  # categorical
                value = random.choice([param_range.min_value, param_range.max_value])
            
            params[param_range.name] = value
        
        return params

class BayesianOptimizer:
    """Bayesian optimization using Gaussian Process."""
    
    def __init__(self, config: OptimizationConfig):
        """Initialize Bayesian optimizer."""
        self.config = config
        self.evaluation_count = 0
        self.best_score = float('-inf')
        self.best_parameters = {}
        self.history = []
        self.X = []  # Parameter vectors
        self.y = []  # Objective values
        
        logger.info("Initialized Bayesian optimizer")
    
    def optimize(self, 
                 parameter_ranges: List[ParameterRange], 
                 objective_function: Callable[[Dict[str, Any]], float]) -> OptimizationResult:
        """Perform Bayesian optimization."""
        try:
            start_time = time.time()
            
            logger.info(f"Bayesian optimization: performing {self.config.max_evaluations} evaluations")
            
            # Initial random evaluations
            n_initial = min(5, self.config.max_evaluations // 4)
            for i in range(n_initial):
                params = self._generate_random_parameters(parameter_ranges)
                score = objective_function(params)
                self._add_evaluation(params, score)
            
            # Bayesian optimization loop
            for i in range(n_initial, self.config.max_evaluations):
                # Generate next parameters using acquisition function
                next_params = self._acquisition_function(parameter_ranges)
                
                # Evaluate
                score = objective_function(next_params)
                self._add_evaluation(next_params, score)
                
                if i % 10 == 0:
                    logger.info(f"Bayesian optimization progress: {i}/{self.config.max_evaluations}")
            
            optimization_time = time.time() - start_time
            
            return OptimizationResult(
                best_parameters=self.best_parameters,
                best_score=self.best_score,
                optimization_history=self.history,
                algorithm_used=OptimizationAlgorithm.BAYESIAN_OPTIMIZATION,
                total_evaluations=self.evaluation_count,
                optimization_time=optimization_time,
                convergence_info={"method": "bayesian_optimization", "acquisition_function": "expected_improvement"}
            )
            
        except Exception as e:
            logger.error(f"Error in Bayesian optimization: {e}")
            raise ClientError(f"Bayesian optimization failed: {str(e)}")
    
    def _add_evaluation(self, params: Dict[str, Any], score: float) -> None:
        """Add evaluation to history and update best."""
        self.evaluation_count += 1
        
        # Convert parameters to vector
        param_vector = self._params_to_vector(params)
        self.X.append(param_vector)
        self.y.append(score)
        
        # Update best
        if score > self.best_score:
            self.best_score = score
            self.best_parameters = params.copy()
        
        # Record history
        self.history.append({
            'evaluation': self.evaluation_count,
            'parameters': params.copy(),
            'score': score,
            'timestamp': time.time()
        })
    
    def _params_to_vector(self, params: Dict[str, Any]) -> List[float]:
        """Convert parameters to numerical vector."""
        # Simple conversion - in practice would need proper normalization
        vector = []
        for value in params.values():
            if isinstance(value, bool):
                vector.append(1.0 if value else 0.0)
            elif isinstance(value, (int, float)):
                vector.append(float(value))
            else:
                vector.append(0.0)  # Fallback
        return vector
    
    def _acquisition_function(self, parameter_ranges: List[ParameterRange]) -> Dict[str, Any]:
        """Generate next parameters using acquisition function."""
        try:
            if len(self.X) < 2:
                return self._generate_random_parameters(parameter_ranges)
            
            # Simplified Expected Improvement acquisition function
            best_candidates = []
            
            # Generate candidate points
            for _ in range(100):
                candidate_params = self._generate_random_parameters(parameter_ranges)
                candidate_vector = self._params_to_vector(candidate_params)
                
                # Calculate expected improvement (simplified)
                mean_prediction = self._predict_mean(candidate_vector)
                std_prediction = self._predict_std(candidate_vector)
                
                # Expected improvement
                improvement = mean_prediction - self.best_score
                z = improvement / (std_prediction + 1e-9)
                
                # Expected improvement formula
                ei = improvement * self._normal_cdf(z) + std_prediction * self._normal_pdf(z)
                
                best_candidates.append((ei, candidate_params))
            
            # Select best candidate
            best_candidates.sort(key=lambda x: x[0], reverse=True)
            return best_candidates[0][1]
            
        except Exception as e:
            logger.error(f"Error in acquisition function: {e}")
            return self._generate_random_parameters(parameter_ranges)
    
    def _predict_mean(self, x: List[float]) -> float:
        """Predict mean using simple interpolation."""
        if not self.X:
            return 0.0
        
        # Simple nearest neighbor prediction
        distances = [np.linalg.norm(np.array(x) - np.array(xi)) for xi in self.X]
        min_idx = np.argmin(distances)
        return self.y[min_idx]
    
    def _predict_std(self, x: List[float]) -> float:
        """Predict standard deviation."""
        if len(self.y) < 2:
            return 1.0
        
        # Simple distance-based uncertainty
        distances = [np.linalg.norm(np.array(x) - np.array(xi)) for xi in self.X]
        min_distance = min(distances)
        return max(0.1, min_distance)
    
    def _normal_cdf(self, x: float) -> float:
        """Normal CDF approximation."""
        return 0.5 * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))
    
    def _normal_pdf(self, x: float) -> float:
        """Normal PDF."""
        return np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)
    
    def _generate_random_parameters(self, parameter_ranges: List[ParameterRange]) -> Dict[str, Any]:
        """Generate random parameters within ranges."""
        params = {}
        
        for param_range in parameter_ranges:
            if param_range.param_type == "float":
                value = random.uniform(param_range.min_value, param_range.max_value)
            elif param_range.param_type == "int":
                value = random.randint(int(param_range.min_value), int(param_range.max_value))
            elif param_range.param_type == "bool":
                value = random.choice([True, False])
            else:  # categorical
                value = random.choice([param_range.min_value, param_range.max_value])
            
            params[param_range.name] = value
        
        return params

class GeneticAlgorithmOptimizer:
    """Genetic algorithm parameter optimizer."""
    
    def __init__(self, config: OptimizationConfig):
        """Initialize genetic algorithm optimizer."""
        self.config = config
        self.evaluation_count = 0
        self.best_score = float('-inf')
        self.best_parameters = {}
        self.history = []
        self.population = []
        self.population_size = 20
        
        if config.random_seed:
            random.seed(config.random_seed)
            np.random.seed(config.random_seed)
        
        logger.info("Initialized Genetic Algorithm optimizer")
    
    def optimize(self, 
                 parameter_ranges: List[ParameterRange], 
                 objective_function: Callable[[Dict[str, Any]], float]) -> OptimizationResult:
        """Perform genetic algorithm optimization."""
        try:
            start_time = time.time()
            
            logger.info(f"Genetic algorithm: performing {self.config.max_evaluations} evaluations")
            
            # Initialize population
            self._initialize_population(parameter_ranges)
            
            generation = 0
            while self.evaluation_count < self.config.max_evaluations:
                # Evaluate population
                fitness_scores = []
                for individual in self.population:
                    score = objective_function(individual)
                    fitness_scores.append(score)
                    self.evaluation_count += 1
                    
                    # Update best
                    if score > self.best_score:
                        self.best_score = score
                        self.best_parameters = individual.copy()
                    
                    # Record history
                    self.history.append({
                        'evaluation': self.evaluation_count,
                        'parameters': individual.copy(),
                        'score': score,
                        'generation': generation,
                        'timestamp': time.time()
                    })
                    
                    if self.evaluation_count >= self.config.max_evaluations:
                        break
                
                if self.evaluation_count >= self.config.max_evaluations:
                    break
                
                # Selection, crossover, mutation
                self._evolve_population(parameter_ranges, fitness_scores)
                generation += 1
                
                if generation % 5 == 0:
                    logger.info(f"Genetic algorithm progress: generation {generation}, evaluations {self.evaluation_count}")
            
            optimization_time = time.time() - start_time
            
            return OptimizationResult(
                best_parameters=self.best_parameters,
                best_score=self.best_score,
                optimization_history=self.history,
                algorithm_used=OptimizationAlgorithm.GENETIC_ALGORITHM,
                total_evaluations=self.evaluation_count,
                optimization_time=optimization_time,
                convergence_info={"method": "genetic_algorithm", "generations": generation}
            )
            
        except Exception as e:
            logger.error(f"Error in genetic algorithm optimization: {e}")
            raise ClientError(f"Genetic algorithm optimization failed: {str(e)}")
    
    def _initialize_population(self, parameter_ranges: List[ParameterRange]) -> None:
        """Initialize random population."""
        self.population = []
        for _ in range(self.population_size):
            individual = self._generate_random_parameters(parameter_ranges)
            self.population.append(individual)
    
    def _evolve_population(self, parameter_ranges: List[ParameterRange], fitness_scores: List[float]) -> None:
        """Evolve population through selection, crossover, and mutation."""
        try:
            # Selection (tournament selection)
            new_population = []
            
            # Keep best individual (elitism)
            best_idx = np.argmax(fitness_scores)
            new_population.append(self.population[best_idx].copy())
            
            # Generate rest through crossover and mutation
            while len(new_population) < self.population_size:
                # Tournament selection
                parent1 = self._tournament_selection(fitness_scores)
                parent2 = self._tournament_selection(fitness_scores)
                
                # Crossover
                child = self._crossover(parent1, parent2, parameter_ranges)
                
                # Mutation
                child = self._mutate(child, parameter_ranges)
                
                new_population.append(child)
            
            self.population = new_population
            
        except Exception as e:
            logger.error(f"Error evolving population: {e}")
    
    def _tournament_selection(self, fitness_scores: List[float], tournament_size: int = 3) -> Dict[str, Any]:
        """Tournament selection."""
        tournament_indices = random.sample(range(len(self.population)), tournament_size)
        tournament_scores = [fitness_scores[i] for i in tournament_indices]
        winner_idx = tournament_indices[np.argmax(tournament_scores)]
        return self.population[winner_idx]
    
    def _crossover(self, parent1: Dict[str, Any], parent2: Dict[str, Any], parameter_ranges: List[ParameterRange]) -> Dict[str, Any]:
        """Uniform crossover."""
        child = {}
        
        for param_range in parameter_ranges:
            name = param_range.name
            if random.random() < 0.5:
                child[name] = parent1[name]
            else:
                child[name] = parent2[name]
        
        return child
    
    def _mutate(self, individual: Dict[str, Any], parameter_ranges: List[ParameterRange], mutation_rate: float = 0.1) -> Dict[str, Any]:
        """Mutate individual."""
        mutated = individual.copy()
        
        for param_range in parameter_ranges:
            if random.random() < mutation_rate:
                # Generate new random value
                if param_range.param_type == "float":
                    mutated[param_range.name] = random.uniform(param_range.min_value, param_range.max_value)
                elif param_range.param_type == "int":
                    mutated[param_range.name] = random.randint(int(param_range.min_value), int(param_range.max_value))
                elif param_range.param_type == "bool":
                    mutated[param_range.name] = random.choice([True, False])
                else:  # categorical
                    mutated[param_range.name] = random.choice([param_range.min_value, param_range.max_value])
        
        return mutated
    
    def _generate_random_parameters(self, parameter_ranges: List[ParameterRange]) -> Dict[str, Any]:
        """Generate random parameters within ranges."""
        params = {}
        
        for param_range in parameter_ranges:
            if param_range.param_type == "float":
                value = random.uniform(param_range.min_value, param_range.max_value)
            elif param_range.param_type == "int":
                value = random.randint(int(param_range.min_value), int(param_range.max_value))
            elif param_range.param_type == "bool":
                value = random.choice([True, False])
            else:  # categorical
                value = random.choice([param_range.min_value, param_range.max_value])
            
            params[param_range.name] = value
        
        return params

class ParameterTuner:
    """Main parameter tuning coordinator."""
    
    def __init__(self, config: OptimizationConfig):
        """Initialize parameter tuner."""
        self.config = config
        self.optimizer = self._create_optimizer()
        
        logger.info(f"Initialized ParameterTuner with {config.algorithm.value} algorithm")
    
    def _create_optimizer(self):
        """Create optimizer based on config."""
        if self.config.algorithm == OptimizationAlgorithm.GRID_SEARCH:
            return GridSearchOptimizer(self.config)
        elif self.config.algorithm == OptimizationAlgorithm.RANDOM_SEARCH:
            return RandomSearchOptimizer(self.config)
        elif self.config.algorithm == OptimizationAlgorithm.BAYESIAN_OPTIMIZATION:
            return BayesianOptimizer(self.config)
        elif self.config.algorithm == OptimizationAlgorithm.GENETIC_ALGORITHM:
            return GeneticAlgorithmOptimizer(self.config)
        else:
            raise ClientError(f"Unsupported optimization algorithm: {self.config.algorithm}")
    
    def optimize_parameters(self, 
                           parameter_ranges: List[ParameterRange], 
                           objective_function: Callable[[Dict[str, Any]], float]) -> OptimizationResult:
        """Optimize parameters using the configured algorithm."""
        try:
            logger.info(f"Starting parameter optimization with {self.config.algorithm.value}")
            logger.info(f"Parameter ranges: {[r.name for r in parameter_ranges]}")
            
            result = self.optimizer.optimize(parameter_ranges, objective_function)
            
            logger.info(f"Optimization completed: best score = {result.best_score:.4f}")
            logger.info(f"Best parameters: {result.best_parameters}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in parameter optimization: {e}")
            raise ClientError(f"Parameter optimization failed: {str(e)}")
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get optimization statistics."""
        if hasattr(self.optimizer, 'history') and self.optimizer.history:
            scores = [h['score'] for h in self.optimizer.history]
            return {
                "total_evaluations": len(self.optimizer.history),
                "best_score": max(scores),
                "worst_score": min(scores),
                "avg_score": np.mean(scores),
                "score_std": np.std(scores),
                "algorithm": self.config.algorithm.value
            }
        else:
            return {"total_evaluations": 0, "algorithm": self.config.algorithm.value}

__all__ = [
    "ParameterTuner",
    "GridSearchOptimizer",
    "RandomSearchOptimizer", 
    "BayesianOptimizer",
    "GeneticAlgorithmOptimizer",
    "OptimizationAlgorithm",
    "ParameterRange",
    "OptimizationResult",
    "OptimizationConfig",
]