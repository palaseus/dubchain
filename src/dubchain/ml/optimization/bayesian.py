"""
Bayesian Optimization for Consensus Parameters

This module provides Bayesian optimization for consensus parameters including:
- Gaussian Process-based parameter optimization
- Acquisition function optimization
- Multi-objective optimization
- Parameter space exploration
- Performance prediction and optimization
- Automated parameter tuning
"""

import logging

logger = logging.getLogger(__name__)
import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
import time
from datetime import datetime, timedelta
import json
import hashlib
from collections import deque

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    from scipy.optimize import minimize
    from scipy.stats import norm
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

from ...errors import BridgeError, ClientError
from ...logging import get_logger

logger = get_logger(__name__)


@dataclass
class ParameterConfig:
    """Configuration for parameter optimization."""
    enable_bayesian_optimization: bool = True
    kernel_type: str = "RBF"  # RBF, Matern, WhiteKernel
    acquisition_function: str = "EI"  # EI (Expected Improvement), UCB (Upper Confidence Bound), PI (Probability of Improvement)
    n_initial_samples: int = 10
    n_iterations: int = 100
    exploration_weight: float = 0.1
    enable_multi_objective: bool = True
    enable_constraint_handling: bool = True
    parameter_bounds: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    optimization_metrics: List[str] = field(default_factory=lambda: ["throughput", "latency", "security"])


@dataclass
class ParameterSpace:
    """Parameter space definition."""
    parameter_name: str
    parameter_type: str  # continuous, discrete, categorical
    bounds: Tuple[float, float]
    default_value: float
    description: str = ""


@dataclass
class OptimizationResult:
    """Result of parameter optimization."""
    best_parameters: Dict[str, float]
    best_score: float
    optimization_history: List[Dict[str, Any]]
    convergence_curve: List[float]
    parameter_importance: Dict[str, float]
    confidence_intervals: Dict[str, Tuple[float, float]]
    timestamp: float = field(default_factory=time.time)


@dataclass
class PerformanceMetrics:
    """Performance metrics for parameter evaluation."""
    throughput: float
    latency: float
    security_score: float
    energy_efficiency: float
    scalability: float
    timestamp: float = field(default_factory=time.time)


class GaussianProcessOptimizer:
    """Gaussian Process-based parameter optimizer."""
    
    def __init__(self, config: ParameterConfig):
        self.config = config
        self.gp_model: Optional[GaussianProcessRegressor] = None
        self.parameter_spaces: List[ParameterSpace] = []
        self.optimization_history: List[Dict[str, Any]] = []
        self.best_parameters: Optional[Dict[str, float]] = None
        self.best_score: float = float('-inf')
        
    def add_parameter_space(self, parameter_space: ParameterSpace) -> None:
        """Add parameter space to optimization."""
        self.parameter_spaces.append(parameter_space)
        self.config.parameter_bounds[parameter_space.parameter_name] = parameter_space.bounds
    
    def initialize_gp_model(self) -> None:
        """Initialize Gaussian Process model."""
        if not SKLEARN_AVAILABLE:
            logger.warning("Scikit-learn not available, skipping GP initialization")
            return
        
        try:
            # Choose kernel based on configuration
            if self.config.kernel_type == "RBF":
                kernel = RBF(length_scale=1.0)
            elif self.config.kernel_type == "Matern":
                kernel = Matern(length_scale=1.0, nu=1.5)
            elif self.config.kernel_type == "WhiteKernel":
                kernel = WhiteKernel(noise_level=0.1)
            else:
                kernel = RBF(length_scale=1.0)
            
            # Initialize GP model
            self.gp_model = GaussianProcessRegressor(
                kernel=kernel,
                alpha=1e-6,
                normalize_y=True,
                n_restarts_optimizer=10
            )
            
            logger.info("Gaussian Process model initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize GP model: {e}")
            raise BridgeError(f"GP model initialization failed: {e}")
    
    def optimize_parameters(self, objective_function: callable) -> OptimizationResult:
        """Optimize parameters using Bayesian optimization."""
        if not self.gp_model:
            self.initialize_gp_model()
        
        if not self.gp_model:
            logger.warning("GP model not available, using random search")
            return self._random_search(objective_function)
        
        try:
            # Initialize with random samples
            initial_samples = self._generate_initial_samples()
            
            # Evaluate initial samples
            for sample in initial_samples:
                score = objective_function(sample)
                self.optimization_history.append({
                    'parameters': sample.copy(),
                    'score': score,
                    'iteration': len(self.optimization_history)
                })
                
                if score > self.best_score:
                    self.best_score = score
                    self.best_parameters = sample.copy()
            
            # Bayesian optimization loop
            for iteration in range(self.config.n_iterations):
                # Fit GP model
                X = np.array([h['parameters'] for h in self.optimization_history])
                y = np.array([h['score'] for h in self.optimization_history])
                
                self.gp_model.fit(X, y)
                
                # Find next point to evaluate using acquisition function
                next_point = self._optimize_acquisition_function()
                
                # Evaluate next point
                score = objective_function(next_point)
                
                # Update history
                self.optimization_history.append({
                    'parameters': next_point.copy(),
                    'score': score,
                    'iteration': len(self.optimization_history)
                })
                
                # Update best parameters
                if score > self.best_score:
                    self.best_score = score
                    self.best_parameters = next_point.copy()
                
                logger.info(f"Iteration {iteration + 1}, Best Score: {self.best_score:.4f}")
            
            # Calculate convergence curve
            convergence_curve = [h['score'] for h in self.optimization_history]
            
            # Calculate parameter importance
            parameter_importance = self._calculate_parameter_importance()
            
            # Calculate confidence intervals
            confidence_intervals = self._calculate_confidence_intervals()
            
            return OptimizationResult(
                best_parameters=self.best_parameters or {},
                best_score=self.best_score,
                optimization_history=self.optimization_history,
                convergence_curve=convergence_curve,
                parameter_importance=parameter_importance,
                confidence_intervals=confidence_intervals
            )
            
        except Exception as e:
            logger.error(f"Failed to optimize parameters: {e}")
            raise BridgeError(f"Parameter optimization failed: {e}")
    
    def _generate_initial_samples(self) -> List[Dict[str, float]]:
        """Generate initial random samples."""
        samples = []
        
        for _ in range(self.config.n_initial_samples):
            sample = {}
            for param_name, bounds in self.config.parameter_bounds.items():
                sample[param_name] = np.random.uniform(bounds[0], bounds[1])
            samples.append(sample)
        
        return samples
    
    def _optimize_acquisition_function(self) -> Dict[str, float]:
        """Optimize acquisition function to find next point."""
        if not SCIPY_AVAILABLE:
            return self._random_sample()
        
        try:
            # Define acquisition function
            def acquisition_function(x):
                x_dict = self._array_to_dict(x)
                return self._evaluate_acquisition_function(x_dict)
            
            # Optimize acquisition function
            result = minimize(
                acquisition_function,
                x0=self._random_sample_array(),
                bounds=list(self.config.parameter_bounds.values()),
                method='L-BFGS-B'
            )
            
            return self._array_to_dict(result.x)
            
        except Exception as e:
            logger.error(f"Failed to optimize acquisition function: {e}")
            return self._random_sample()
    
    def _evaluate_acquisition_function(self, parameters: Dict[str, float]) -> float:
        """Evaluate acquisition function."""
        if not self.gp_model:
            return 0.0
        
        try:
            # Convert to array
            x = self._dict_to_array(parameters)
            x = x.reshape(1, -1)
            
            # Get GP predictions
            mean, std = self.gp_model.predict(x, return_std=True)
            
            # Calculate acquisition function value
            if self.config.acquisition_function == "EI":
                # Expected Improvement
                if std[0] > 0:
                    z = (mean[0] - self.best_score) / std[0]
                    ei = (mean[0] - self.best_score) * norm.cdf(z) + std[0] * norm.pdf(z)
                else:
                    ei = 0.0
                return -ei  # Minimize negative EI
            
            elif self.config.acquisition_function == "UCB":
                # Upper Confidence Bound
                ucb = mean[0] + self.config.exploration_weight * std[0]
                return -ucb  # Minimize negative UCB
            
            elif self.config.acquisition_function == "PI":
                # Probability of Improvement
                if std[0] > 0:
                    z = (mean[0] - self.best_score) / std[0]
                    pi = norm.cdf(z)
                else:
                    pi = 0.0
                return -pi  # Minimize negative PI
            
            else:
                return 0.0
                
        except Exception as e:
            logger.error(f"Failed to evaluate acquisition function: {e}")
            return 0.0
    
    def _random_sample(self) -> Dict[str, float]:
        """Generate random sample."""
        sample = {}
        for param_name, bounds in self.config.parameter_bounds.items():
            sample[param_name] = np.random.uniform(bounds[0], bounds[1])
        return sample
    
    def _random_sample_array(self) -> np.ndarray:
        """Generate random sample as array."""
        sample = []
        for bounds in self.config.parameter_bounds.values():
            sample.append(np.random.uniform(bounds[0], bounds[1]))
        return np.array(sample)
    
    def _dict_to_array(self, parameters: Dict[str, float]) -> np.ndarray:
        """Convert parameter dictionary to array."""
        array = []
        for param_name in self.config.parameter_bounds.keys():
            array.append(parameters.get(param_name, 0.0))
        return np.array(array)
    
    def _array_to_dict(self, array: np.ndarray) -> Dict[str, float]:
        """Convert array to parameter dictionary."""
        parameters = {}
        for i, param_name in enumerate(self.config.parameter_bounds.keys()):
            parameters[param_name] = array[i]
        return parameters
    
    def _calculate_parameter_importance(self) -> Dict[str, float]:
        """Calculate parameter importance using GP model."""
        if not self.gp_model or not self.optimization_history:
            return {}
        
        try:
            # Get parameter names
            param_names = list(self.config.parameter_bounds.keys())
            
            # Calculate importance using partial dependence
            importance = {}
            for i, param_name in enumerate(param_names):
                # Sample parameter space
                param_values = np.linspace(
                    self.config.parameter_bounds[param_name][0],
                    self.config.parameter_bounds[param_name][1],
                    20
                )
                
                # Calculate variance for this parameter
                variances = []
                for val in param_values:
                    # Create sample with this parameter fixed
                    sample = self._random_sample()
                    sample[param_name] = val
                    
                    # Get prediction variance
                    x = self._dict_to_array(sample).reshape(1, -1)
                    _, std = self.gp_model.predict(x, return_std=True)
                    variances.append(std[0])
                
                # Importance is proportional to variance
                importance[param_name] = np.mean(variances)
            
            # Normalize importance
            total_importance = sum(importance.values())
            if total_importance > 0:
                for param_name in importance:
                    importance[param_name] /= total_importance
            
            return importance
            
        except Exception as e:
            logger.error(f"Failed to calculate parameter importance: {e}")
            return {}
    
    def _calculate_confidence_intervals(self) -> Dict[str, Tuple[float, float]]:
        """Calculate confidence intervals for parameters."""
        if not self.optimization_history:
            return {}
        
        try:
            # Get parameter names
            param_names = list(self.config.parameter_bounds.keys())
            
            # Calculate confidence intervals
            confidence_intervals = {}
            for param_name in param_names:
                # Get parameter values from history
                values = [h['parameters'][param_name] for h in self.optimization_history]
                
                # Calculate 95% confidence interval
                mean_val = np.mean(values)
                std_val = np.std(values)
                
                # Use normal distribution approximation
                ci_lower = mean_val - 1.96 * std_val
                ci_upper = mean_val + 1.96 * std_val
                
                confidence_intervals[param_name] = (ci_lower, ci_upper)
            
            return confidence_intervals
            
        except Exception as e:
            logger.error(f"Failed to calculate confidence intervals: {e}")
            return {}
    
    def _random_search(self, objective_function: callable) -> OptimizationResult:
        """Fallback random search optimization."""
        logger.info("Using random search optimization")
        
        best_score = float('-inf')
        best_parameters = {}
        
        for iteration in range(self.config.n_iterations):
            # Generate random sample
            sample = self._random_sample()
            
            # Evaluate objective function
            score = objective_function(sample)
            
            # Update best
            if score > best_score:
                best_score = score
                best_parameters = sample.copy()
            
            # Store in history
            self.optimization_history.append({
                'parameters': sample.copy(),
                'score': score,
                'iteration': iteration
            })
        
        return OptimizationResult(
            best_parameters=best_parameters,
            best_score=best_score,
            optimization_history=self.optimization_history,
            convergence_curve=[h['score'] for h in self.optimization_history],
            parameter_importance={},
            confidence_intervals={}
        )


class MultiObjectiveOptimizer:
    """Multi-objective parameter optimizer."""
    
    def __init__(self, config: ParameterConfig):
        self.config = config
        self.objectives: List[str] = config.optimization_metrics
        self.pareto_front: List[Dict[str, Any]] = []
        self.optimization_history: List[Dict[str, Any]] = []
    
    def optimize_multi_objective(self, objective_functions: Dict[str, callable]) -> List[OptimizationResult]:
        """Optimize multiple objectives simultaneously."""
        try:
            # Initialize GP models for each objective
            gp_models = {}
            for objective in self.objectives:
                if objective in objective_functions:
                    gp_models[objective] = GaussianProcessOptimizer(self.config)
            
            # Multi-objective optimization loop
            for iteration in range(self.config.n_iterations):
                # Generate candidate solutions
                candidates = self._generate_candidates()
                
                # Evaluate all objectives for each candidate
                for candidate in candidates:
                    scores = {}
                    for objective, func in objective_functions.items():
                        scores[objective] = func(candidate)
                    
                    # Store in history
                    self.optimization_history.append({
                        'parameters': candidate.copy(),
                        'scores': scores.copy(),
                        'iteration': iteration
                    })
                    
                    # Update Pareto front
                    self._update_pareto_front(candidate, scores)
                
                logger.info(f"Multi-objective iteration {iteration + 1}, Pareto front size: {len(self.pareto_front)}")
            
            # Convert Pareto front to optimization results
            results = []
            for pareto_point in self.pareto_front:
                # Calculate composite score
                composite_score = self._calculate_composite_score(pareto_point['scores'])
                
                result = OptimizationResult(
                    best_parameters=pareto_point['parameters'],
                    best_score=composite_score,
                    optimization_history=self.optimization_history,
                    convergence_curve=[h['scores'] for h in self.optimization_history],
                    parameter_importance={},
                    confidence_intervals={}
                )
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to optimize multi-objective: {e}")
            return []
    
    def _generate_candidates(self) -> List[Dict[str, float]]:
        """Generate candidate solutions."""
        candidates = []
        
        # Generate random candidates
        for _ in range(10):
            candidate = {}
            for param_name, bounds in self.config.parameter_bounds.items():
                candidate[param_name] = np.random.uniform(bounds[0], bounds[1])
            candidates.append(candidate)
        
        return candidates
    
    def _update_pareto_front(self, candidate: Dict[str, float], scores: Dict[str, float]) -> None:
        """Update Pareto front with new candidate."""
        # Check if candidate dominates any existing Pareto points
        dominated_indices = []
        
        for i, pareto_point in enumerate(self.pareto_front):
            if self._dominates(scores, pareto_point['scores']):
                dominated_indices.append(i)
        
        # Remove dominated points
        for i in reversed(dominated_indices):
            self.pareto_front.pop(i)
        
        # Check if candidate is dominated by any Pareto point
        is_dominated = False
        for pareto_point in self.pareto_front:
            if self._dominates(pareto_point['scores'], scores):
                is_dominated = True
                break
        
        # Add candidate to Pareto front if not dominated
        if not is_dominated:
            self.pareto_front.append({
                'parameters': candidate.copy(),
                'scores': scores.copy()
            })
    
    def _dominates(self, scores1: Dict[str, float], scores2: Dict[str, float]) -> bool:
        """Check if scores1 dominates scores2."""
        # scores1 dominates scores2 if it's better in at least one objective
        # and not worse in any objective
        
        better_in_at_least_one = False
        
        for objective in self.objectives:
            if objective in scores1 and objective in scores2:
                if scores1[objective] > scores2[objective]:
                    better_in_at_least_one = True
                elif scores1[objective] < scores2[objective]:
                    return False  # scores1 is worse in this objective
        
        return better_in_at_least_one
    
    def _calculate_composite_score(self, scores: Dict[str, float]) -> float:
        """Calculate composite score from multiple objectives."""
        # Weighted sum of objectives
        weights = {
            'throughput': 0.3,
            'latency': -0.3,  # Negative because lower latency is better
            'security': 0.2,
            'energy_efficiency': 0.1,
            'scalability': 0.1
        }
        
        composite_score = 0.0
        for objective, score in scores.items():
            weight = weights.get(objective, 0.1)
            composite_score += weight * score
        
        return composite_score


class ConsensusParameterOptimizer:
    """Main consensus parameter optimizer."""
    
    def __init__(self, config: ParameterConfig):
        self.config = config
        self.gp_optimizer = GaussianProcessOptimizer(config)
        self.multi_objective_optimizer = MultiObjectiveOptimizer(config)
        self.optimization_history = deque(maxlen=1000)
        
    def add_consensus_parameter(self, parameter_space: ParameterSpace) -> None:
        """Add consensus parameter to optimization."""
        self.gp_optimizer.add_parameter_space(parameter_space)
    
    def optimize_consensus_parameters(self, performance_evaluator: callable) -> OptimizationResult:
        """Optimize consensus parameters."""
        try:
            # Define objective function
            def objective_function(parameters: Dict[str, float]) -> float:
                # Evaluate performance with given parameters
                metrics = performance_evaluator(parameters)
                
                # Calculate composite score
                score = self._calculate_performance_score(metrics)
                
                # Store in history
                self.optimization_history.append({
                    'parameters': parameters.copy(),
                    'metrics': metrics,
                    'score': score,
                    'timestamp': time.time()
                })
                
                return score
            
            # Run optimization
            result = self.gp_optimizer.optimize_parameters(objective_function)
            
            logger.info(f"Consensus parameter optimization completed. Best score: {result.best_score:.4f}")
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to optimize consensus parameters: {e}")
            raise BridgeError(f"Consensus parameter optimization failed: {e}")
    
    def optimize_multi_objective_consensus(self, performance_evaluator: callable) -> List[OptimizationResult]:
        """Optimize consensus parameters for multiple objectives."""
        try:
            # Define objective functions
            objective_functions = {}
            
            for metric in self.config.optimization_metrics:
                def make_objective_func(metric_name):
                    def objective_func(parameters: Dict[str, float]) -> float:
                        metrics = performance_evaluator(parameters)
                        return getattr(metrics, metric_name, 0.0)
                    return objective_func
                
                objective_functions[metric] = make_objective_func(metric)
            
            # Run multi-objective optimization
            results = self.multi_objective_optimizer.optimize_multi_objective(objective_functions)
            
            logger.info(f"Multi-objective consensus optimization completed. Found {len(results)} Pareto optimal solutions")
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to optimize multi-objective consensus: {e}")
            return []
    
    def _calculate_performance_score(self, metrics: PerformanceMetrics) -> float:
        """Calculate composite performance score."""
        # Weighted combination of metrics
        weights = {
            'throughput': 0.3,
            'latency': -0.3,  # Negative because lower latency is better
            'security_score': 0.2,
            'energy_efficiency': 0.1,
            'scalability': 0.1
        }
        
        score = 0.0
        score += weights['throughput'] * metrics.throughput
        score += weights['latency'] * metrics.latency
        score += weights['security_score'] * metrics.security_score
        score += weights['energy_efficiency'] * metrics.energy_efficiency
        score += weights['scalability'] * metrics.scalability
        
        return score
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get optimization statistics."""
        return {
            "total_optimizations": len(self.optimization_history),
            "bayesian_optimization_enabled": self.config.enable_bayesian_optimization,
            "multi_objective_enabled": self.config.enable_multi_objective,
            "kernel_type": self.config.kernel_type,
            "acquisition_function": self.config.acquisition_function,
            "n_iterations": self.config.n_iterations,
            "parameter_spaces": len(self.gp_optimizer.parameter_spaces),
            "sklearn_available": SKLEARN_AVAILABLE,
            "scipy_available": SCIPY_AVAILABLE,
            "torch_available": TORCH_AVAILABLE
        }
