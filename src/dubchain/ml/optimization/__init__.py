"""
ML Parameter Optimization Module

This module provides Bayesian optimization for consensus parameters including:
- Gaussian Process-based parameter optimization
- Acquisition function optimization
- Multi-objective optimization
- Parameter space exploration
- Performance prediction and optimization
- Automated parameter tuning
"""

from .bayesian import (
    ConsensusParameterOptimizer,
    ParameterConfig,
    GaussianProcessOptimizer,
    MultiObjectiveOptimizer,
    ParameterSpace,
    OptimizationResult,
    PerformanceMetrics,
)

__all__ = [
    "ConsensusParameterOptimizer",
    "ParameterConfig",
    "GaussianProcessOptimizer",
    "MultiObjectiveOptimizer",
    "ParameterSpace",
    "OptimizationResult",
    "PerformanceMetrics",
]