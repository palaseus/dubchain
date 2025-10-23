"""
Base hardware acceleration framework for DubChain.

This module provides the base classes and interfaces for all hardware
acceleration implementations.
"""

import logging

logger = logging.getLogger(__name__)
import time
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from .detection import AccelerationType, HardwareCapabilities


class AccelerationStatus(Enum):
    """Status of hardware acceleration."""
    
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    INITIALIZING = "initializing"
    ERROR = "error"


@dataclass
class PerformanceMetrics:
    """Performance metrics for hardware acceleration."""
    
    operations_count: int = 0
    total_time_ms: float = 0.0
    avg_time_ms: float = 0.0
    min_time_ms: float = float('inf')
    max_time_ms: float = 0.0
    throughput_ops_per_sec: float = 0.0
    memory_usage_mb: float = 0.0
    gpu_utilization_percent: float = 0.0
    cpu_utilization_percent: float = 0.0
    error_count: int = 0
    last_updated: float = field(default_factory=time.time)


@dataclass
class AccelerationResult:
    """Result of hardware acceleration operation."""
    
    success: bool
    execution_time: float
    throughput: float
    data: Any = None
    error_message: Optional[str] = None
    memory_used: float = 0.0
    acceleration_type: Optional[AccelerationType] = None


@dataclass
class AccelerationConfig:
    """Configuration for hardware acceleration."""
    
    enable_acceleration: bool = True
    fallback_to_cpu: bool = True
    max_memory_mb: int = 1024
    batch_size: int = 1000
    timeout_ms: int = 5000
    retry_count: int = 3
    enable_profiling: bool = True
    log_level: str = "INFO"
    custom_params: Dict[str, Any] = field(default_factory=dict)


class HardwareAccelerator(ABC):
    """Abstract base class for hardware accelerators."""
    
    def __init__(self, config: AccelerationConfig):
        """Initialize hardware accelerator."""
        self.config = config
        self.status = AccelerationStatus.UNAVAILABLE
        self.metrics = PerformanceMetrics()
        self._lock = threading.Lock()
        self._initialized = False
        
    @abstractmethod
    def initialize(self) -> bool:
        """Initialize the hardware accelerator."""
        pass
    
    @abstractmethod
    def cleanup(self) -> None:
        """Cleanup resources."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if acceleration is available."""
        pass
    
    @abstractmethod
    def get_acceleration_type(self) -> AccelerationType:
        """Get the acceleration type."""
        pass
    
    @abstractmethod
    def get_device_info(self) -> Dict[str, Any]:
        """Get device information."""
        pass
    
    @abstractmethod
    def execute_operation(self, operation: str, data: Dict[str, Any]) -> 'AccelerationResult':
        """Execute an operation with given data."""
        pass
    
    def start_profiling(self) -> None:
        """Start performance profiling."""
        if self.config.enable_profiling:
            self.metrics.last_updated = time.time()
    
    def stop_profiling(self) -> None:
        """Stop performance profiling."""
        if self.config.enable_profiling:
            self._update_metrics()
    
    def _update_metrics(self) -> None:
        """Update performance metrics."""
        with self._lock:
            if self.metrics.operations_count > 0:
                self.metrics.avg_time_ms = (
                    self.metrics.total_time_ms / self.metrics.operations_count
                )
                self.metrics.throughput_ops_per_sec = (
                    self.metrics.operations_count * 1000.0 / self.metrics.total_time_ms
                )
            self.metrics.last_updated = time.time()
    
    def _record_operation(self, duration_ms: float, success: bool = True) -> None:
        """Record an operation for metrics."""
        with self._lock:
            self.metrics.operations_count += 1
            self.metrics.total_time_ms += duration_ms
            self.metrics.min_time_ms = min(self.metrics.min_time_ms, duration_ms)
            self.metrics.max_time_ms = max(self.metrics.max_time_ms, duration_ms)
            
            if not success:
                self.metrics.error_count += 1
    
    def get_metrics(self) -> PerformanceMetrics:
        """Get current performance metrics."""
        with self._lock:
            return PerformanceMetrics(
                operations_count=self.metrics.operations_count,
                total_time_ms=self.metrics.total_time_ms,
                avg_time_ms=self.metrics.avg_time_ms,
                min_time_ms=self.metrics.min_time_ms,
                max_time_ms=self.metrics.max_time_ms,
                throughput_ops_per_sec=self.metrics.throughput_ops_per_sec,
                memory_usage_mb=self.metrics.memory_usage_mb,
                gpu_utilization_percent=self.metrics.gpu_utilization_percent,
                cpu_utilization_percent=self.metrics.cpu_utilization_percent,
                error_count=self.metrics.error_count,
                last_updated=self.metrics.last_updated,
            )
    
    def reset_metrics(self) -> None:
        """Reset performance metrics."""
        with self._lock:
            self.metrics = PerformanceMetrics()
    
    def __enter__(self):
        """Context manager entry."""
        if not self._initialized:
            self.initialize()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()
    
    # Test compatibility methods
    def get_name(self) -> str:
        """Get accelerator name."""
        return self.get_acceleration_type().value
    
    def get_performance_score(self) -> float:
        """Get performance score."""
        metrics = self.get_metrics()
        return metrics.throughput_ops_per_sec
    
    def benchmark(self) -> Dict[str, Any]:
        """Benchmark the accelerator."""
        return {
            "throughput": self.get_metrics().throughput_ops_per_sec,
            "latency": self.get_metrics().avg_time_ms
        }
    
    def allocate_memory(self, size: int) -> Any:
        """Allocate memory."""
        return self.memory_manager.allocate(size) if self.memory_manager else None
    
    def free_memory(self, memory: Any) -> None:
        """Free memory."""
        if self.memory_manager:
            self.memory_manager.deallocate(memory)


class BatchProcessor(ABC):
    """Abstract base class for batch processing operations."""
    
    @abstractmethod
    def process_batch(self, data: List[Any]) -> List[Any]:
        """Process a batch of data."""
        pass
    
    @abstractmethod
    def get_optimal_batch_size(self) -> int:
        """Get optimal batch size for this processor."""
        pass


class MemoryManager(ABC):
    """Abstract base class for memory management."""
    
    @abstractmethod
    def allocate(self, size_bytes: int) -> Any:
        """Allocate memory."""
        pass
    
    @abstractmethod
    def deallocate(self, memory: Any) -> None:
        """Deallocate memory."""
        pass
    
    @abstractmethod
    def get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        pass
    
    @abstractmethod
    def get_available_memory(self) -> float:
        """Get available memory in MB."""
        pass


class ErrorHandler:
    """Error handling utilities for hardware acceleration."""
    
    @staticmethod
    def handle_error(error: Exception, context: str = "") -> None:
        """Handle hardware acceleration errors."""
        error_msg = f"Hardware acceleration error in {context}: {str(error)}"
        logger.info(f"ERROR: {error_msg}")
        
        # Log error details
        import traceback
        traceback.print_exc()
    
    @staticmethod
    def should_fallback(error: Exception) -> bool:
        """Determine if should fallback to CPU."""
        # Common errors that should trigger fallback
        fallback_errors = [
            "CUDA out of memory",
            "OpenCL out of memory", 
            "Metal out of memory",
            "device not found",
            "driver not found",
            "not supported",
        ]
        
        error_str = str(error).lower()
        return any(fallback_error in error_str for fallback_error in fallback_errors)


class FallbackManager:
    """Manages fallback between different acceleration methods."""
    
    def __init__(self, accelerators: List[HardwareAccelerator]):
        """Initialize fallback manager."""
        self.accelerators = accelerators
        self.current_accelerator_index = 0
        self._lock = threading.Lock()
    
    def get_current_accelerator(self) -> Optional[HardwareAccelerator]:
        """Get current accelerator."""
        with self._lock:
            if self.current_accelerator_index < len(self.accelerators):
                return self.accelerators[self.current_accelerator_index]
            return None
    
    def fallback_to_next(self) -> bool:
        """Fallback to next available accelerator."""
        with self._lock:
            self.current_accelerator_index += 1
            if self.current_accelerator_index < len(self.accelerators):
                return True
            return False
    
    def reset_to_first(self) -> None:
        """Reset to first accelerator."""
        with self._lock:
            self.current_accelerator_index = 0
    
    def get_all_available(self) -> List[HardwareAccelerator]:
        """Get all available accelerators."""
        return [acc for acc in self.accelerators if acc.is_available()]
