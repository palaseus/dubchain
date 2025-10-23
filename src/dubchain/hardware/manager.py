"""
Hardware manager for DubChain.

This module provides unified hardware acceleration management including:
- Automatic hardware detection and selection
- Fallback management between acceleration types
- Performance monitoring and optimization
- Resource management
"""

import logging

logger = logging.getLogger(__name__)
import time
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from .detection import HardwareDetector, HardwareCapabilities, AccelerationType
from .base import (
    HardwareAccelerator,
    AccelerationConfig,
    AccelerationResult,
    FallbackManager,
    ErrorHandler,
)
from .cuda import CUDAAccelerator, CUDAConfig
from .opencl import OpenCLAccelerator, OpenCLConfig
from .cpu import CPUAccelerator, CPUConfig


@dataclass
class HardwareManagerConfig:
    """Configuration for hardware manager."""
    
    enable_cuda: bool = True
    enable_opencl: bool = True
    enable_cpu_simd: bool = True
    auto_detect_hardware: bool = True
    fallback_enabled: bool = True
    performance_monitoring: bool = True
    max_memory_usage_mb: int = 2048
    enable_profiling: bool = True
    log_level: str = "INFO"
    custom_configs: Dict[str, Any] = field(default_factory=dict)


class HardwareManager:
    """Unified hardware acceleration manager."""
    
    def __init__(self, config: Optional[HardwareManagerConfig] = None):
        """Initialize hardware manager."""
        self.config = config or HardwareManagerConfig()
        self.detector = HardwareDetector()
        self.capabilities: Optional[HardwareCapabilities] = None
        self.accelerators: Dict[str, HardwareAccelerator] = {}
        self.fallback_manager: Optional[FallbackManager] = None
        self.current_accelerator: Optional[HardwareAccelerator] = None
        self._lock = threading.Lock()
        self._initialized = False
        
        # Performance tracking
        self.total_operations = 0
        self.total_time_ms = 0.0
        self.fallback_count = 0
        
    def initialize(self) -> bool:
        """Initialize hardware manager."""
        if self._initialized:
            return True
        
        try:
            # Detect hardware capabilities
            self.capabilities = self.detector.detect_all()
            
            # Initialize accelerators based on capabilities
            self._initialize_accelerators()
            
            # Setup fallback manager
            if self.config.fallback_enabled and self.accelerators:
                self.fallback_manager = FallbackManager(list(self.accelerators.values()))
                self.current_accelerator = self.fallback_manager.get_current_accelerator()
            
            self._initialized = True
            return True
            
        except Exception as e:
            ErrorHandler.handle_error(e, "Hardware manager initialization")
            return False
    
    def _initialize_accelerators(self) -> None:
        """Initialize available accelerators."""
        if not self.capabilities:
            return
        
        # Initialize CUDA accelerator
        if (self.config.enable_cuda and 
            AccelerationType.CUDA in self.capabilities.available_accelerations):
            try:
                cuda_config = CUDAConfig(
                    enable_acceleration=True,
                    fallback_to_cpu=self.config.fallback_enabled,
                    max_memory_mb=self.config.max_memory_usage_mb,
                    enable_profiling=self.config.enable_profiling,
                )
                cuda_accelerator = CUDAAccelerator(cuda_config)
                if cuda_accelerator.initialize():
                    self.accelerators["cuda"] = cuda_accelerator
            except Exception as e:
                ErrorHandler.handle_error(e, "CUDA accelerator initialization")
        
        # Initialize OpenCL accelerator
        if (self.config.enable_opencl and 
            AccelerationType.OPENCL in self.capabilities.available_accelerations):
            try:
                opencl_config = OpenCLConfig(
                    enable_acceleration=True,
                    fallback_to_cpu=self.config.fallback_enabled,
                    max_memory_mb=self.config.max_memory_usage_mb,
                    enable_profiling=self.config.enable_profiling,
                )
                opencl_accelerator = OpenCLAccelerator(opencl_config)
                if opencl_accelerator.initialize():
                    self.accelerators["opencl"] = opencl_accelerator
            except Exception as e:
                ErrorHandler.handle_error(e, "OpenCL accelerator initialization")
        
        # Initialize CPU accelerator
        if self.config.enable_cpu_simd:
            try:
                cpu_config = CPUConfig(
                    enable_acceleration=True,
                    fallback_to_cpu=False,  # CPU is the fallback
                    max_memory_mb=self.config.max_memory_usage_mb,
                    enable_profiling=self.config.enable_profiling,
                )
                cpu_accelerator = CPUAccelerator(cpu_config)
                if cpu_accelerator.initialize():
                    self.accelerators["cpu"] = cpu_accelerator
            except Exception as e:
                ErrorHandler.handle_error(e, "CPU accelerator initialization")
    
    def cleanup(self) -> None:
        """Cleanup all accelerators."""
        with self._lock:
            for accelerator in self.accelerators.values():
                try:
                    accelerator.cleanup()
                except Exception as e:
                    ErrorHandler.handle_error(e, f"Cleaning up {accelerator.get_acceleration_type().value}")
            
            self.accelerators.clear()
            self.current_accelerator = None
            self.fallback_manager = None
            self._initialized = False
    
    def get_current_accelerator(self) -> Optional[HardwareAccelerator]:
        """Get current accelerator."""
        return self.current_accelerator
    
    def get_available_accelerators(self) -> List[str]:
        """Get all available accelerators."""
        return [name for name, acc in self.accelerators.items() if acc.is_available()]
    
    def get_accelerator_by_type(self, acc_type: AccelerationType) -> Optional[HardwareAccelerator]:
        """Get accelerator by type."""
        for accelerator in self.accelerators.values():
            if accelerator.get_acceleration_type() == acc_type and accelerator.is_available():
                return accelerator
        return None
    
    def switch_accelerator(self, acc_type: AccelerationType) -> bool:
        """Switch to specific accelerator type."""
        accelerator = self.get_accelerator_by_type(acc_type)
        if accelerator:
            with self._lock:
                self.current_accelerator = accelerator
            return True
        return False
    
    def hash_batch(self, data: List[bytes], algorithm: str = "sha256") -> List[bytes]:
        """Hash batch of data using current accelerator."""
        if not self._initialized:
            if not self.initialize():
                raise RuntimeError("Hardware manager not initialized")
        
        if not self.current_accelerator:
            raise RuntimeError("No accelerator available")
        
        start_time = time.time()
        try:
            results = self.current_accelerator.hash_batch(data, algorithm)
            duration_ms = (time.time() - start_time) * 1000
            
            with self._lock:
                self.total_operations += 1
                self.total_time_ms += duration_ms
            
            return results
            
        except Exception as e:
            # Try fallback if available
            if self.config.fallback_enabled and self.fallback_manager:
                if self.fallback_manager.fallback_to_next():
                    self.current_accelerator = self.fallback_manager.get_current_accelerator()
                    self.fallback_count += 1
                    
                    # Retry with fallback accelerator
                    try:
                        results = self.current_accelerator.hash_batch(data, algorithm)
                        duration_ms = (time.time() - start_time) * 1000
                        
                        with self._lock:
                            self.total_operations += 1
                            self.total_time_ms += duration_ms
                        
                        return results
                    except Exception as fallback_error:
                        ErrorHandler.handle_error(fallback_error, "Fallback accelerator failed")
            
            ErrorHandler.handle_error(e, "Hash batch operation")
            raise
    
    def verify_signatures_batch(self, signatures: List[bytes], messages: List[bytes]) -> List[bool]:
        """Verify batch of signatures using current accelerator."""
        if not self._initialized:
            if not self.initialize():
                raise RuntimeError("Hardware manager not initialized")
        
        if not self.current_accelerator:
            raise RuntimeError("No accelerator available")
        
        start_time = time.time()
        try:
            results = self.current_accelerator.verify_signatures_batch(signatures, messages)
            duration_ms = (time.time() - start_time) * 1000
            
            with self._lock:
                self.total_operations += 1
                self.total_time_ms += duration_ms
            
            return results
            
        except Exception as e:
            # Try fallback if available
            if self.config.fallback_enabled and self.fallback_manager:
                if self.fallback_manager.fallback_to_next():
                    self.current_accelerator = self.fallback_manager.get_current_accelerator()
                    self.fallback_count += 1
                    
                    # Retry with fallback accelerator
                    try:
                        results = self.current_accelerator.verify_signatures_batch(signatures, messages)
                        duration_ms = (time.time() - start_time) * 1000
                        
                        with self._lock:
                            self.total_operations += 1
                            self.total_time_ms += duration_ms
                        
                        return results
                    except Exception as fallback_error:
                        ErrorHandler.handle_error(fallback_error, "Fallback accelerator failed")
            
            ErrorHandler.handle_error(e, "Signature verification operation")
            raise
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        stats = {
            "manager": {
                "initialized": self._initialized,
                "total_operations": self.total_operations,
                "total_time_ms": self.total_time_ms,
                "avg_time_ms": self.total_time_ms / max(self.total_operations, 1),
                "fallback_count": self.fallback_count,
                "available_accelerators": len(self.get_available_accelerators()),
            },
            "capabilities": self.capabilities.__dict__ if self.capabilities else {},
            "accelerators": [],
        }
        
        # Get stats from each accelerator
        for accelerator in self.accelerators.values():
            try:
                acc_stats = accelerator.get_performance_stats()
                acc_stats["type"] = accelerator.get_acceleration_type().value
                acc_stats["is_current"] = accelerator == self.current_accelerator
                stats["accelerators"].append(acc_stats)
            except Exception as e:
                ErrorHandler.handle_error(e, f"Getting stats for {accelerator.get_acceleration_type().value}")
        
        return stats
    
    def get_recommended_acceleration(self) -> Optional[AccelerationType]:
        """Get recommended acceleration type."""
        if self.capabilities:
            return self.capabilities.recommended_acceleration
        return None
    
    def benchmark_accelerators(self, test_data: Optional[List[bytes]] = None, iterations: int = 10) -> Dict[str, Any]:
        """Benchmark all available accelerators."""
        if not self._initialized:
            if not self.initialize():
                raise RuntimeError("Hardware manager not initialized")
        
        # Use default test data if none provided
        if test_data is None:
            test_data = [b"test_data_" + str(i).encode() for i in range(100)]
        
        results = {}
        
        for accelerator_name in self.get_available_accelerators():
            accelerator = self.accelerators[accelerator_name]
            acc_type = accelerator.get_acceleration_type().value
            times = []
            
            try:
                # Warm up
                accelerator.hash_batch(test_data[:10])
                
                # Benchmark
                for _ in range(iterations):
                    start_time = time.time()
                    accelerator.hash_batch(test_data)
                    duration_ms = (time.time() - start_time) * 1000
                    times.append(duration_ms)
                
                results[acc_type] = {
                    "avg_time_ms": sum(times) / len(times),
                    "min_time_ms": min(times),
                    "max_time_ms": max(times),
                    "throughput_ops_per_sec": len(test_data) * 1000.0 / (sum(times) / len(times)),
                    "throughput": len(test_data) * 1000.0 / (sum(times) / len(times)),
                    "latency": sum(times) / len(times),
                    "success": True,
                }
                
            except Exception as e:
                ErrorHandler.handle_error(e, f"Benchmarking {acc_type}")
                results[acc_type] = {
                    "error": str(e),
                    "success": False,
                }
        
        return results
    
    def optimize_configuration(self) -> Dict[str, Any]:
        """Optimize accelerator configuration based on performance."""
        if not self._initialized:
            return {}
        
        # Run benchmarks
        test_data = [b"test_data_" + str(i).encode() for i in range(1000)]
        benchmark_results = self.benchmark_accelerators(test_data)
        
        # Find best performing accelerator
        best_accelerator = None
        best_throughput = 0.0
        
        for acc_type, results in benchmark_results.items():
            if results.get("success", False):
                throughput = results.get("throughput_ops_per_sec", 0.0)
                if throughput > best_throughput:
                    best_throughput = throughput
                    best_accelerator = acc_type
        
        # Switch to best accelerator
        if best_accelerator:
            acc_type = AccelerationType(best_accelerator)
            self.switch_accelerator(acc_type)
        
        return {
            "best_accelerator": best_accelerator,
            "best_throughput": best_throughput,
            "benchmark_results": benchmark_results,
            "current_accelerator": self.current_accelerator.get_acceleration_type().value if self.current_accelerator else None,
        }
    
    def __enter__(self):
        """Context manager entry."""
        self.initialize()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()
    
    # Test compatibility methods
    def register_accelerator(self, name: str, accelerator: HardwareAccelerator) -> None:
        """Register an accelerator."""
        self.accelerators[name] = accelerator
    
    def execute_operation(self, accelerator_name: str, operation: str, params: Dict[str, Any]) -> AccelerationResult:
        """Execute an operation on a specific accelerator."""
        if accelerator_name not in self.accelerators:
            raise ValueError(f"Accelerator {accelerator_name} not found")
        
        accelerator = self.accelerators[accelerator_name]
        if not accelerator.is_available():
            raise RuntimeError(f"Accelerator {accelerator_name} not available")
        
        start_time = time.time()
        try:
            # Use the accelerator's execute_operation method
            result = accelerator.execute_operation(operation, params)
            
            return result
            
        except MemoryError as e:
            # Re-raise memory errors as they are critical
            raise e
        except Exception as e:
            execution_time = time.time() - start_time
            ErrorHandler.handle_error(e, f"Executing {operation} on {accelerator_name}")
            return AccelerationResult(
                success=False,
                execution_time=execution_time,
                throughput=0.0,
                error_message=str(e),
                acceleration_type=accelerator.get_acceleration_type()
            )
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get system statistics."""
        return {
            "total_accelerators": len(self.accelerators),
            "available_accelerators": len([acc for acc in self.accelerators.values() if acc.is_available()]),
            "current_accelerator": self.current_accelerator.get_acceleration_type().value if self.current_accelerator else None,
            "system_capabilities": self.detector.detect_all() if self.detector else {}
        }
    
    def select_best_accelerator(self) -> Optional[str]:
        """Select the best available accelerator."""
        available_accelerators = [(name, acc) for name, acc in self.accelerators.items() if acc.is_available()]
        if not available_accelerators:
            return None
        
        # Simple selection based on acceleration type priority
        priority_order = [
            AccelerationType.CUDA,
            AccelerationType.METAL,
            AccelerationType.OPENCL,
            AccelerationType.AVX512,
            AccelerationType.AVX2,
            AccelerationType.AVX,
            AccelerationType.NEON,
            AccelerationType.SSE4,
            AccelerationType.CPU,
            AccelerationType.CPU_MULTICORE,
        ]
        
        for acc_type in priority_order:
            for name, accelerator in available_accelerators:
                if accelerator.get_acceleration_type() == acc_type:
                    return name
        
        # Fallback to first available
        return available_accelerators[0][0]
