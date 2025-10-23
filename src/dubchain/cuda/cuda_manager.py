"""
CUDA Manager for DubChain.

This module provides centralized CUDA management throughout the entire codebase,
ensuring consistent GPU acceleration across all components.
"""

import logging

logger = logging.getLogger(__name__)
import time
import threading
from typing import Dict, Any, Optional, List, Callable, Union
from contextlib import contextmanager

from .cuda_config import CUDAConfig, get_global_cuda_config
from .cuda_utils import (
    cuda_available,
    get_cuda_device,
    get_cuda_memory_info,
    cuda_synchronize,
    cuda_memory_cleanup,
    cuda_memory_usage,
    cuda_device_info,
)


class CUDAManager:
    """
    Centralized CUDA manager for DubChain.
    
    Provides unified GPU acceleration across all components including:
    - Cryptography operations
    - Consensus mechanisms
    - Sharding operations
    - Networking operations
    - Storage operations
    - Testing and benchmarking
    """
    
    def __init__(self, config: Optional[CUDAConfig] = None):
        """Initialize CUDA manager."""
        self.config = config or get_global_cuda_config()
        self.device = get_cuda_device()
        self.available = cuda_available()
        
        # Performance tracking
        self.performance_metrics = {
            'total_operations': 0,
            'gpu_operations': 0,
            'cpu_fallbacks': 0,
            'batch_operations': 0,
            'avg_gpu_time': 0.0,
            'avg_cpu_time': 0.0,
            'memory_usage': 0,
            'peak_memory_usage': 0,
        }
        
        # Thread safety
        self._metrics_lock = threading.Lock()
        self._operation_lock = threading.Lock()
        
        # Initialize CUDA if available
        if self.available:
            self._initialize_cuda()
        
        logger.info(f"🚀 CUDA Manager initialized - Available: {self.available}, Device: {self.device}")
    
    def _initialize_cuda(self):
        """Initialize CUDA environment."""
        try:
            # Set memory limits
            if self.config.memory_limit_mb > 0:
                self._set_memory_limits()
            
            # Initialize memory pool if enabled
            if self.config.enable_memory_pool:
                self._initialize_memory_pool()
            
            # Log initialization
            if self.config.enable_cuda_logging:
                self._log_cuda_info()
                
        except Exception as e:
            logger.info(f"Warning: CUDA initialization failed: {e}")
            self.available = False
    
    def _set_memory_limits(self):
        """Set CUDA memory limits."""
        try:
            import torch
            if torch.cuda.is_available():
                # Set memory fraction
                torch.cuda.set_per_process_memory_fraction(self.config.memory_fraction)
                
                # Set memory limit
                memory_limit_bytes = self.config.memory_limit_mb * 1024 * 1024
                torch.cuda.set_per_process_memory_limit(memory_limit_bytes)
        except Exception as e:
            logger.info(f"Warning: Failed to set CUDA memory limits: {e}")
    
    def _initialize_memory_pool(self):
        """Initialize CUDA memory pool."""
        try:
            import cupy as cp
            if cp.cuda.is_available():
                # Enable memory pool
                cp.cuda.set_allocator(cp.cuda.MemoryPool().malloc)
        except Exception as e:
            logger.info(f"Warning: Failed to initialize CUDA memory pool: {e}")
    
    def _log_cuda_info(self):
        """Log CUDA information."""
        device_info = cuda_device_info()
        memory_info = get_cuda_memory_info()
        
        logger.info(f"CUDA Device: {device_info.get('name', 'Unknown')}")
        logger.info(f"Compute Capability: {device_info.get('compute_capability', 'Unknown')}")
        logger.info(f"Total Memory: {memory_info.get('total_memory', 0) / (1024**3):.2f} GB")
        logger.info(f"Free Memory: {memory_info.get('free_memory', 0) / (1024**3):.2f} GB")
    
    def should_use_gpu(self, algorithm: str, batch_size: int = 1) -> bool:
        """Determine if GPU should be used for a given algorithm and batch size."""
        return self.config.should_use_gpu(algorithm, batch_size) and self.available
    
    def execute_gpu_operation(self, 
                            operation_func: Callable,
                            data: Union[Any, List[Any]],
                            algorithm: str = "general",
                            fallback_func: Optional[Callable] = None) -> Any:
        """
        Execute operation with GPU acceleration and automatic fallback.
        
        Args:
            operation_func: GPU-accelerated operation function
            data: Data to process
            algorithm: Algorithm type for configuration
            fallback_func: CPU fallback function (optional)
            
        Returns:
            Operation result
        """
        batch_size = len(data) if isinstance(data, list) else 1
        
        if not self.should_use_gpu(algorithm, batch_size):
            # Use CPU fallback
            if fallback_func:
                return self._execute_cpu_operation(fallback_func, data)
            else:
                return self._execute_cpu_operation(operation_func, data)
        
        # Try GPU operation
        try:
            return self._execute_gpu_operation(operation_func, data)
        except Exception as e:
            if self.config.fallback_to_cpu:
                logger.info(f"⚠️  GPU operation failed, falling back to CPU: {e}")
                if fallback_func:
                    return self._execute_cpu_operation(fallback_func, data)
                else:
                    return self._execute_cpu_operation(operation_func, data)
            else:
                raise
    
    def _execute_gpu_operation(self, operation_func: Callable, data: Union[Any, List[Any]]) -> Any:
        """Execute GPU operation with performance tracking."""
        start_time = time.time()
        
        with self._operation_lock:
            try:
                result = operation_func(data)
                cuda_synchronize()
                
                # Update metrics
                operation_time = time.time() - start_time
                self._update_metrics(operation_time, True, len(data) if isinstance(data, list) else 1)
                
                return result
            except Exception as e:
                # Update metrics for failed operation
                operation_time = time.time() - start_time
                self._update_metrics(operation_time, True, len(data) if isinstance(data, list) else 1)
                raise
    
    def _execute_cpu_operation(self, operation_func: Callable, data: Union[Any, List[Any]]) -> Any:
        """Execute CPU operation with performance tracking."""
        start_time = time.time()
        
        try:
            result = operation_func(data)
            
            # Update metrics
            operation_time = time.time() - start_time
            self._update_metrics(operation_time, False, len(data) if isinstance(data, list) else 1)
            
            return result
        except Exception as e:
            # Update metrics for failed operation
            operation_time = time.time() - start_time
            self._update_metrics(operation_time, False, len(data) if isinstance(data, list) else 1)
            raise
    
    def _update_metrics(self, operation_time: float, is_gpu: bool, operation_count: int):
        """Update performance metrics."""
        with self._metrics_lock:
            self.performance_metrics['total_operations'] += operation_count
            
            if is_gpu:
                self.performance_metrics['gpu_operations'] += operation_count
                self.performance_metrics['batch_operations'] += 1
                
                # Update average GPU time
                total_gpu_ops = self.performance_metrics['gpu_operations']
                current_avg = self.performance_metrics['avg_gpu_time']
                self.performance_metrics['avg_gpu_time'] = (
                    (current_avg * (total_gpu_ops - operation_count) + operation_time) / total_gpu_ops
                )
            else:
                self.performance_metrics['cpu_fallbacks'] += operation_count
                
                # Update average CPU time
                total_cpu_ops = self.performance_metrics['cpu_fallbacks']
                current_avg = self.performance_metrics['avg_cpu_time']
                self.performance_metrics['avg_cpu_time'] = (
                    (current_avg * (total_cpu_ops - operation_count) + operation_time) / total_cpu_ops
                )
            
            # Update memory usage
            if self.available:
                memory_usage = cuda_memory_usage()
                if memory_usage.get('available'):
                    self.performance_metrics['memory_usage'] = memory_usage.get('allocated', 0)
                    self.performance_metrics['peak_memory_usage'] = max(
                        self.performance_metrics['peak_memory_usage'],
                        memory_usage.get('allocated', 0)
                    )
    
    def batch_operation(self, 
                       operation_func: Callable,
                       data_list: List[Any],
                       algorithm: str = "general",
                       fallback_func: Optional[Callable] = None) -> List[Any]:
        """
        Execute batch operation with GPU acceleration.
        
        Args:
            operation_func: GPU-accelerated batch operation function
            data_list: List of data to process
            algorithm: Algorithm type for configuration
            fallback_func: CPU fallback function (optional)
            
        Returns:
            List of operation results
        """
        if not data_list:
            return []
        
        batch_size = len(data_list)
        
        if not self.should_use_gpu(algorithm, batch_size):
            # Use CPU fallback
            if fallback_func:
                return self._execute_cpu_batch_operation(fallback_func, data_list)
            else:
                return self._execute_cpu_batch_operation(operation_func, data_list)
        
        # Try GPU batch operation
        try:
            return self._execute_gpu_batch_operation(operation_func, data_list)
        except Exception as e:
            if self.config.fallback_to_cpu:
                logger.info(f"⚠️  GPU batch operation failed, falling back to CPU: {e}")
                if fallback_func:
                    return self._execute_cpu_batch_operation(fallback_func, data_list)
                else:
                    return self._execute_cpu_batch_operation(operation_func, data_list)
            else:
                raise
    
    def _execute_gpu_batch_operation(self, operation_func: Callable, data_list: List[Any]) -> List[Any]:
        """Execute GPU batch operation."""
        # Process in chunks to manage memory
        chunk_size = self.config.chunk_size
        results = []
        
        for i in range(0, len(data_list), chunk_size):
            chunk = data_list[i:i + chunk_size]
            chunk_result = self._execute_gpu_operation(operation_func, chunk)
            
            if isinstance(chunk_result, list):
                results.extend(chunk_result)
            else:
                results.append(chunk_result)
        
        return results
    
    def _execute_cpu_batch_operation(self, operation_func: Callable, data_list: List[Any]) -> List[Any]:
        """Execute CPU batch operation."""
        results = []
        for data in data_list:
            result = self._execute_cpu_operation(operation_func, data)
            results.append(result)
        return results
    
    @contextmanager
    def gpu_context(self, algorithm: str = "general"):
        """Context manager for GPU operations."""
        if not self.available or not self.config.enable_cuda:
            yield False
            return
        
        try:
            yield True
        finally:
            if self.config.enable_cuda_logging and self.config.log_memory_usage:
                self._log_memory_usage()
    
    def _log_memory_usage(self):
        """Log current memory usage."""
        memory_usage = cuda_memory_usage()
        if memory_usage.get('available'):
            logger.info(f"CUDA Memory - Allocated: {memory_usage.get('allocated', 0) / (1024**2):.2f} MB, "
                  f"Reserved: {memory_usage.get('reserved', 0) / (1024**2):.2f} MB")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics."""
        with self._metrics_lock:
            total_ops = self.performance_metrics['total_operations']
            gpu_utilization = 0.0
            if total_ops > 0:
                gpu_utilization = self.performance_metrics['gpu_operations'] / total_ops
            
            return {
                **self.performance_metrics,
                'gpu_utilization': gpu_utilization,
                'cuda_available': self.available,
                'device': self.device,
                'config': self.config.to_dict(),
            }
    
    def reset_metrics(self):
        """Reset performance metrics."""
        with self._metrics_lock:
            self.performance_metrics = {
                'total_operations': 0,
                'gpu_operations': 0,
                'cpu_fallbacks': 0,
                'batch_operations': 0,
                'avg_gpu_time': 0.0,
                'avg_cpu_time': 0.0,
                'memory_usage': 0,
                'peak_memory_usage': 0,
            }
    
    def cleanup(self):
        """Clean up CUDA resources."""
        if self.available:
            cuda_memory_cleanup()
            cuda_synchronize()
    
    def benchmark_operation(self, 
                          gpu_func: Callable,
                          cpu_func: Callable,
                          test_data: List[Any],
                          algorithm: str = "general",
                          num_iterations: int = 10) -> Dict[str, Any]:
        """
        Benchmark GPU vs CPU operation performance.
        
        Args:
            gpu_func: GPU-accelerated function
            cpu_func: CPU function
            test_data: Test data
            algorithm: Algorithm type
            num_iterations: Number of benchmark iterations
            
        Returns:
            Benchmark results
        """
        logger.info(f"🔬 Benchmarking {algorithm} operation...")
        logger.info(f"   Data size: {len(test_data)} items")
        logger.info(f"   Iterations: {num_iterations}")
        
        # CPU benchmark
        logger.info("   Testing CPU performance...")
        cpu_times = []
        for _ in range(num_iterations):
            start_time = time.time()
            cpu_func(test_data)
            cpu_times.append(time.time() - start_time)
        
        cpu_avg_time = sum(cpu_times) / len(cpu_times)
        cpu_throughput = len(test_data) / cpu_avg_time
        
        # GPU benchmark
        logger.info("   Testing GPU performance...")
        gpu_times = []
        gpu_success = False
        
        if self.should_use_gpu(algorithm, len(test_data)):
            try:
                for _ in range(num_iterations):
                    start_time = time.time()
                    gpu_func(test_data)
                    cuda_synchronize()
                    gpu_times.append(time.time() - start_time)
                gpu_success = True
            except Exception as e:
                logger.info(f"   GPU benchmark failed: {e}")
        
        if gpu_success and gpu_times:
            gpu_avg_time = sum(gpu_times) / len(gpu_times)
            gpu_throughput = len(test_data) / gpu_avg_time
            speedup = cpu_avg_time / gpu_avg_time
        else:
            gpu_avg_time = 0
            gpu_throughput = 0
            speedup = 0
        
        results = {
            'algorithm': algorithm,
            'data_size': len(test_data),
            'iterations': num_iterations,
            'cpu_avg_time': cpu_avg_time,
            'gpu_avg_time': gpu_avg_time,
            'cpu_throughput': cpu_throughput,
            'gpu_throughput': gpu_throughput,
            'speedup': speedup,
            'gpu_success': gpu_success,
            'used_gpu': self.should_use_gpu(algorithm, len(test_data)),
        }
        
        logger.info(f"   Results:")
        logger.info(f"     CPU time: {cpu_avg_time:.4f}s")
        logger.info(f"     GPU time: {gpu_avg_time:.4f}s")
        logger.info(f"     Speedup: {speedup:.2f}x")
        logger.info(f"     GPU success: {gpu_success}")
        
        return results


# Global CUDA manager instance
_global_cuda_manager: Optional[CUDAManager] = None


def get_global_cuda_manager() -> CUDAManager:
    """Get the global CUDA manager."""
    global _global_cuda_manager
    if _global_cuda_manager is None:
        _global_cuda_manager = CUDAManager()
    return _global_cuda_manager


def set_global_cuda_manager(manager: CUDAManager) -> None:
    """Set the global CUDA manager."""
    global _global_cuda_manager
    _global_cuda_manager = manager


def reset_global_cuda_manager() -> None:
    """Reset the global CUDA manager."""
    global _global_cuda_manager
    if _global_cuda_manager is not None:
        _global_cuda_manager.cleanup()
    _global_cuda_manager = None
