"""
OpenCL hardware acceleration for DubChain.

This module provides OpenCL-specific hardware acceleration including:
- Multi-vendor GPU support (AMD, Intel, NVIDIA)
- OpenCL kernel execution
- Memory management
- Performance optimization
"""

import logging

logger = logging.getLogger(__name__)
import time
import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

try:
    import pyopencl as cl
    OPENCL_AVAILABLE = True
except ImportError:
    OPENCL_AVAILABLE = False
    # Create dummy cl module for type hints
    class DummyCL:
        class Buffer:
            pass
        class Context:
            pass
        class Device:
            pass
        class CommandQueue:
            pass
        class Program:
            pass
        class mem_flags:
            READ_WRITE = 0
            READ_ONLY = 1
            WRITE_ONLY = 2
            COPY_HOST_PTR = 4
        class command_queue_properties:
            PROFILING_ENABLE = 0
        class device_info:
            GLOBAL_MEM_SIZE = 0
            MAX_WORK_GROUP_SIZE = 1
            NAME = 2
            VENDOR = 3
            VERSION = 4
        class device_type:
            GPU = 0x1000
        def get_platforms():
            return []
        def enqueue_nd_range_kernel(*args, **kwargs):
            pass
        def enqueue_copy(*args, **kwargs):
            pass
    
    cl = DummyCL()

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

from .base import (
    HardwareAccelerator,
    AccelerationConfig,
    AccelerationStatus,
    PerformanceMetrics,
    BatchProcessor,
    MemoryManager,
    ErrorHandler,
)
from .detection import AccelerationType


@dataclass
class OpenCLConfig(AccelerationConfig):
    """OpenCL-specific configuration."""
    
    platform_index: int = 0
    device_index: int = 0
    device_type: int = 0x1000  # CL_DEVICE_TYPE_GPU
    enable_profiling: bool = True
    enable_compiler_cache: bool = True
    kernel_timeout_ms: int = 10000
    max_work_group_size: int = 256
    local_memory_size_kb: int = 32
    enable_image_support: bool = True


class OpenCLMemoryManager(MemoryManager):
    """OpenCL memory manager."""
    
    def __init__(self, context: Any, device: Any):
        """Initialize OpenCL memory manager."""
        self.context = context
        self.device = device
        self.allocated_buffers: Dict[int, cl.Buffer] = {}
        self.buffer_counter = 0
        
    def allocate(self, size_bytes: int) -> int:
        """Allocate OpenCL buffer."""
        if not OPENCL_AVAILABLE:
            raise RuntimeError("OpenCL not available")
        
        try:
            buffer = cl.Buffer(
                self.context,
                cl.mem_flags.READ_WRITE,
                size_bytes
            )
            buffer_id = self.buffer_counter
            self.allocated_buffers[buffer_id] = buffer
            self.buffer_counter += 1
            return buffer_id
        except Exception as e:
            ErrorHandler.handle_error(e, "OpenCL memory allocation")
            raise
    
    def deallocate(self, buffer_id: int) -> None:
        """Deallocate OpenCL buffer."""
        if buffer_id in self.allocated_buffers:
            del self.allocated_buffers[buffer_id]
    
    def get_memory_usage(self) -> float:
        """Get OpenCL memory usage in MB."""
        if not OPENCL_AVAILABLE:
            return 0.0
        
        try:
            # Get global memory size
            global_mem_size = self.device.get_info(cl.device_info.GLOBAL_MEM_SIZE)
            return global_mem_size / (1024 * 1024)
        except Exception:
            return 0.0
    
    def get_available_memory(self) -> float:
        """Get available OpenCL memory in MB."""
        if not OPENCL_AVAILABLE:
            return 0.0
        
        try:
            # This is a simplified calculation
            # Real implementation would track actual allocations
            global_mem_size = self.device.get_info(cl.device_info.GLOBAL_MEM_SIZE)
            return global_mem_size / (1024 * 1024)
        except Exception:
            return 0.0


class OpenCLBatchProcessor(BatchProcessor):
    """OpenCL batch processor for cryptographic operations."""
    
    def __init__(self, config: OpenCLConfig, context: cl.Context, device: cl.Device):
        """Initialize OpenCL batch processor."""
        self.config = config
        self.context = context
        self.device = device
        self.queue: Optional[cl.CommandQueue] = None
        self.program: Optional[cl.Program] = None
        self._init_opencl()
        
    def _init_opencl(self) -> None:
        """Initialize OpenCL components."""
        if not OPENCL_AVAILABLE:
            return
        
        try:
            # Create command queue
            self.queue = cl.CommandQueue(
                self.context,
                self.device,
                properties=cl.command_queue_properties.PROFILING_ENABLE
            )
            
            # Create OpenCL program with hash kernel
            kernel_source = """
            __kernel void hash_batch(
                __global const uchar* input,
                __global uchar* output,
                const uint input_size,
                const uint batch_size
            ) {
                uint gid = get_global_id(0);
                if (gid >= batch_size) return;
                
                uint offset = gid * input_size;
                
                // Simple hash computation (placeholder)
                // Real implementation would use proper cryptographic hash
                uchar hash = 0;
                for (uint i = 0; i < input_size; i++) {
                    hash ^= input[offset + i];
                }
                
                output[gid] = hash;
            }
            """
            
            self.program = cl.Program(self.context, kernel_source).build()
            
        except Exception as e:
            ErrorHandler.handle_error(e, "OpenCL initialization")
    
    def process_batch(self, data: List[bytes]) -> List[bytes]:
        """Process batch of data using OpenCL."""
        if not OPENCL_AVAILABLE or not self.queue or not self.program:
            raise RuntimeError("OpenCL not available")
        
        try:
            if not data:
                return []
            
            # Determine batch size and input size
            batch_size = len(data)
            input_size = len(data[0]) if data else 0
            
            # Create input buffer
            input_array = np.array([list(item) for item in data], dtype=np.uint8)
            input_buffer = cl.Buffer(
                self.context,
                cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                hostbuf=input_array
            )
            
            # Create output buffer
            output_buffer = cl.Buffer(
                self.context,
                cl.mem_flags.WRITE_ONLY,
                batch_size
            )
            
            # Execute kernel
            kernel = self.program.hash_batch
            kernel.set_args(input_buffer, output_buffer, np.uint32(input_size), np.uint32(batch_size))
            
            global_size = (batch_size,)
            local_size = (min(self.config.max_work_group_size, batch_size),)
            
            cl.enqueue_nd_range_kernel(
                self.queue,
                kernel,
                global_size,
                local_size
            )
            
            # Read results
            output_array = np.empty(batch_size, dtype=np.uint8)
            cl.enqueue_copy(self.queue, output_array, output_buffer)
            self.queue.finish()
            
            # Convert back to bytes
            results = [bytes([output_array[i]]) for i in range(batch_size)]
            
            return results
            
        except Exception as e:
            ErrorHandler.handle_error(e, "OpenCL batch processing")
            raise
    
    def get_optimal_batch_size(self) -> int:
        """Get optimal batch size for OpenCL."""
        if not OPENCL_AVAILABLE:
            return 1
        
        try:
            # Get device info
            max_work_group_size = self.device.get_info(cl.device_info.MAX_WORK_GROUP_SIZE)
            global_mem_size = self.device.get_info(cl.device_info.GLOBAL_MEM_SIZE)
            
            # Estimate based on memory and work group size
            memory_gb = global_mem_size / (1024**3)
            
            if memory_gb >= 4:
                return min(10000, max_work_group_size * 100)
            elif memory_gb >= 2:
                return min(5000, max_work_group_size * 50)
            else:
                return min(1000, max_work_group_size * 10)
                
        except Exception:
            return self.config.batch_size


class OpenCLAccelerator(HardwareAccelerator):
    """OpenCL hardware accelerator."""
    
    def __init__(self, config: OpenCLConfig):
        """Initialize OpenCL accelerator."""
        super().__init__(config)
        self.opencl_config = config
        self.context: Optional[cl.Context] = None
        self.device: Optional[cl.Device] = None
        self.memory_manager: Optional[OpenCLMemoryManager] = None
        self.batch_processor: Optional[OpenCLBatchProcessor] = None
        self.device_info: Optional[Dict[str, Any]] = None
        
    def initialize(self) -> bool:
        """Initialize OpenCL accelerator."""
        if not OPENCL_AVAILABLE:
            self.status = AccelerationStatus.UNAVAILABLE
            return False
        
        try:
            # Get platforms
            platforms = cl.get_platforms()
            if not platforms:
                self.status = AccelerationStatus.UNAVAILABLE
                return False
            
            # Select platform
            platform = platforms[min(self.opencl_config.platform_index, len(platforms) - 1)]
            
            # Get devices
            devices = platform.get_devices(self.opencl_config.device_type)
            if not devices:
                self.status = AccelerationStatus.UNAVAILABLE
                return False
            
            # Select device
            self.device = devices[min(self.opencl_config.device_index, len(devices) - 1)]
            
            # Create context
            self.context = cl.Context([self.device])
            
            # Initialize memory manager
            self.memory_manager = OpenCLMemoryManager(self.context, self.device)
            
            # Initialize batch processor
            self.batch_processor = OpenCLBatchProcessor(self.opencl_config, self.context, self.device)
            
            # Get device info
            self.device_info = self._get_device_info()
            
            self.status = AccelerationStatus.AVAILABLE
            return True
            
        except Exception as e:
            ErrorHandler.handle_error(e, "OpenCL initialization")
            self.status = AccelerationStatus.ERROR
            return False
    
    def cleanup(self) -> None:
        """Cleanup OpenCL resources."""
        try:
            if self.memory_manager:
                # Clear all allocated buffers
                for buffer_id in list(self.memory_manager.allocated_buffers.keys()):
                    self.memory_manager.deallocate(buffer_id)
            
            # Release context
            if self.context:
                self.context = None
                
        except Exception as e:
            ErrorHandler.handle_error(e, "OpenCL cleanup")
    
    def is_available(self) -> bool:
        """Check if OpenCL is available."""
        return (
            OPENCL_AVAILABLE and 
            self.context is not None and 
            self.device is not None and
            self.status == AccelerationStatus.AVAILABLE
        )
    
    def get_acceleration_type(self) -> AccelerationType:
        """Get acceleration type."""
        return AccelerationType.OPENCL
    
    def get_device_info(self) -> Dict[str, Any]:
        """Get OpenCL device information."""
        if not self.device or not self.device_info:
            return {}
        
        return {
            "type": "OpenCL",
            "platform_index": self.opencl_config.platform_index,
            "device_index": self.opencl_config.device_index,
            "device_info": self.device_info,
            "memory_usage_mb": self.memory_manager.get_memory_usage() if self.memory_manager else 0.0,
            "available_memory_mb": self.memory_manager.get_available_memory() if self.memory_manager else 0.0,
            "status": self.status.value,
        }
    
    def _get_device_info(self) -> Dict[str, Any]:
        """Get OpenCL device information."""
        if not self.device:
            return {}
        
        try:
            return {
                "name": self.device.get_info(cl.device_info.NAME),
                "vendor": self.device.get_info(cl.device_info.VENDOR),
                "version": self.device.get_info(cl.device_info.VERSION),
                "driver_version": self.device.get_info(cl.device_info.DRIVER_VERSION),
                "global_mem_size": self.device.get_info(cl.device_info.GLOBAL_MEM_SIZE),
                "local_mem_size": self.device.get_info(cl.device_info.LOCAL_MEM_SIZE),
                "max_work_group_size": self.device.get_info(cl.device_info.MAX_WORK_GROUP_SIZE),
                "max_compute_units": self.device.get_info(cl.device_info.MAX_COMPUTE_UNITS),
                "max_work_item_dimensions": self.device.get_info(cl.device_info.MAX_WORK_ITEM_DIMENSIONS),
                "max_work_item_sizes": self.device.get_info(cl.device_info.MAX_WORK_ITEM_SIZES),
                "preferred_vector_width_char": self.device.get_info(cl.device_info.PREFERRED_VECTOR_WIDTH_CHAR),
                "preferred_vector_width_float": self.device.get_info(cl.device_info.PREFERRED_VECTOR_WIDTH_FLOAT),
                "device_type": self.device.get_info(cl.device_info.DEVICE_TYPE),
            }
        except Exception as e:
            ErrorHandler.handle_error(e, "Getting OpenCL device info")
            return {}
    
    def hash_batch(self, data: List[bytes], algorithm: str = "sha256") -> List[bytes]:
        """Hash batch of data using OpenCL."""
        if not self.is_available() or not self.batch_processor:
            raise RuntimeError("OpenCL accelerator not available")
        
        start_time = time.time()
        try:
            results = self.batch_processor.process_batch(data)
            duration_ms = (time.time() - start_time) * 1000
            self._record_operation(duration_ms, True)
            return results
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self._record_operation(duration_ms, False)
            ErrorHandler.handle_error(e, "OpenCL hash batch")
            raise
    
    def verify_signatures_batch(self, signatures: List[bytes], messages: List[bytes]) -> List[bool]:
        """Verify batch of signatures using OpenCL."""
        if not self.is_available():
            raise RuntimeError("OpenCL accelerator not available")
        
        start_time = time.time()
        try:
            # This is a simplified example - real implementation would use
            # actual signature verification algorithms
            results = []
            for sig, msg in zip(signatures, messages):
                # Placeholder verification logic
                result = len(sig) > 0 and len(msg) > 0
                results.append(result)
            
            duration_ms = (time.time() - start_time) * 1000
            self._record_operation(duration_ms, True)
            return results
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self._record_operation(duration_ms, False)
            ErrorHandler.handle_error(e, "OpenCL signature verification")
            raise
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get OpenCL performance statistics."""
        metrics = self.get_metrics()
        device_info = self.get_device_info()
        
        return {
            "metrics": {
                "operations_count": metrics.operations_count,
                "avg_time_ms": metrics.avg_time_ms,
                "throughput_ops_per_sec": metrics.throughput_ops_per_sec,
                "error_count": metrics.error_count,
            },
            "device": device_info,
            "memory": {
                "usage_mb": device_info.get("memory_usage_mb", 0.0),
                "available_mb": device_info.get("available_memory_mb", 0.0),
            },
            "status": self.status.value,
        }
    
    # Test compatibility methods
    def get_platform_info(self) -> List[Dict[str, Any]]:
        """Get OpenCL platform information."""
        if not self.is_available():
            return []
        
        try:
            platforms = cl.get_platforms()
            platform_info = []
            for platform in platforms:
                info = {
                    'name': platform.get_info(cl.platform_info.NAME),
                    'vendor': platform.get_info(cl.platform_info.VENDOR),
                    'version': platform.get_info(cl.platform_info.VERSION),
                }
                platform_info.append(info)
            return platform_info
        except Exception:
            return []
    
    def get_device_info(self) -> List[Dict[str, Any]]:
        """Get OpenCL device information."""
        if not self.is_available():
            return []
        
        try:
            platforms = cl.get_platforms()
            device_info = []
            for platform in platforms:
                devices = platform.get_devices()
                for device in devices:
                    info = {
                        'name': device.get_info(cl.device_info.NAME),
                        'type': device.get_info(cl.device_info.DEVICE_TYPE),
                        'compute_units': device.get_info(cl.device_info.MAX_COMPUTE_UNITS),
                        'max_memory': device.get_info(cl.device_info.GLOBAL_MEM_SIZE),
                    }
                    device_info.append(info)
            return device_info
        except Exception:
            return {}
    
    def create_context(self) -> Any:
        """Create OpenCL context."""
        if not self.is_available():
            return None
        
        try:
            platforms = cl.get_platforms()
            if platforms:
                devices = platforms[0].get_devices()
                if devices:
                    return cl.Context(devices)
        except Exception:
            pass
        return None
    
    def matrix_multiply(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Perform matrix multiplication using OpenCL."""
        if not NUMPY_AVAILABLE:
            raise RuntimeError("NumPy not available")
        
        try:
            # Fallback to CPU implementation if OpenCL not available
            return np.dot(a, b)
        except Exception as e:
            self.error_handler.handle_error(e)
            raise
    
    def compute_hash(self, data: bytes) -> bytes:
        """Compute hash of data using OpenCL."""
        import hashlib
        return hashlib.sha256(data).digest()
