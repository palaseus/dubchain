"""
CUDA Utilities for DubChain.

This module provides utility functions for CUDA operations throughout
the entire codebase, ensuring consistent GPU acceleration support.
"""

import logging

logger = logging.getLogger(__name__)
import os
import time
import threading
from typing import Optional, Dict, Any, Tuple, List
from contextlib import contextmanager

try:
    import torch
    import torch.cuda
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False


# Global CUDA state
_cuda_available: Optional[bool] = None
_cuda_device: Optional[str] = None
_cuda_memory_info: Optional[Dict[str, Any]] = None
_cuda_lock = threading.Lock()


def cuda_available() -> bool:
    """Check if CUDA is available."""
    global _cuda_available
    
    with _cuda_lock:
        if _cuda_available is None:
            _cuda_available = _check_cuda_availability()
    
    return _cuda_available


def _check_cuda_availability() -> bool:
    """Check CUDA availability across all backends."""
    # Check PyTorch CUDA
    if TORCH_AVAILABLE and torch.cuda.is_available():
        return True
    
    # Check CuPy CUDA
    if CUPY_AVAILABLE and cp.cuda.is_available():
        return True
    
    return False


def get_cuda_device() -> Optional[str]:
    """Get the current CUDA device."""
    global _cuda_device
    
    with _cuda_lock:
        if _cuda_device is None:
            try:
                _cuda_device = _detect_cuda_device()
            except Exception as e:
                logger.info(f"Warning: CUDA device detection failed: {e}")
                _cuda_device = None
    
    return _cuda_device


def _detect_cuda_device() -> Optional[str]:
    """Detect the best available CUDA device."""
    # Check CUDA availability without using the cached version to avoid deadlock
    cuda_available_now = False
    
    # Check PyTorch CUDA
    if TORCH_AVAILABLE and torch.cuda.is_available():
        cuda_available_now = True
        try:
            device_id = torch.cuda.current_device()
            return f"cuda:{device_id}"
        except Exception as e:
            logger.info(f"Warning: PyTorch CUDA device detection failed: {e}")
    
    # Fallback to CuPy
    if CUPY_AVAILABLE and cp.cuda.is_available():
        cuda_available_now = True
        return "cupy"
    
    return None


def get_cuda_memory_info() -> Dict[str, Any]:
    """Get CUDA memory information."""
    global _cuda_memory_info
    
    with _cuda_lock:
        if _cuda_memory_info is None:
            _cuda_memory_info = _get_memory_info()
    
    return _cuda_memory_info


def _get_memory_info() -> Dict[str, Any]:
    """Get detailed CUDA memory information."""
    memory_info = {
        'available': False,
        'total_memory': 0,
        'free_memory': 0,
        'used_memory': 0,
        'device_count': 0,
        'device_name': None,
        'compute_capability': None,
    }
    
    if not cuda_available():
        return memory_info
    
    try:
        if TORCH_AVAILABLE and torch.cuda.is_available():
            device_id = torch.cuda.current_device()
            memory_info.update({
                'available': True,
                'total_memory': torch.cuda.get_device_properties(device_id).total_memory,
                'free_memory': torch.cuda.memory_reserved(device_id) - torch.cuda.memory_allocated(device_id),
                'used_memory': torch.cuda.memory_allocated(device_id),
                'device_count': torch.cuda.device_count(),
                'device_name': torch.cuda.get_device_name(device_id),
                'compute_capability': torch.cuda.get_device_capability(device_id),
            })
        elif CUPY_AVAILABLE and cp.cuda.is_available():
            mempool = cp.get_default_memory_pool()
            memory_info.update({
                'available': True,
                'total_memory': cp.cuda.runtime.memGetInfo()[1],
                'free_memory': cp.cuda.runtime.memGetInfo()[0],
                'used_memory': cp.cuda.runtime.memGetInfo()[1] - cp.cuda.runtime.memGetInfo()[0],
                'device_count': cp.cuda.runtime.getDeviceCount(),
                'device_name': 'CuPy Device',
                'compute_capability': None,
            })
    except Exception as e:
        logger.info(f"Warning: Failed to get CUDA memory info: {e}")
    
    return memory_info


def cuda_synchronize() -> None:
    """Synchronize CUDA operations."""
    if not cuda_available():
        return
    
    try:
        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.synchronize()
        elif CUPY_AVAILABLE and cp.cuda.is_available():
            cp.cuda.Stream.null.synchronize()
    except Exception as e:
        logger.info(f"Warning: CUDA synchronization failed: {e}")


def cuda_memory_cleanup() -> None:
    """Clean up CUDA memory."""
    if not cuda_available():
        return
    
    try:
        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif CUPY_AVAILABLE and cp.cuda.is_available():
            cp.get_default_memory_pool().free_all_blocks()
    except Exception as e:
        logger.info(f"Warning: CUDA memory cleanup failed: {e}")


@contextmanager
def cuda_context(device: Optional[str] = None):
    """Context manager for CUDA operations."""
    if not cuda_available():
        yield None
        return
    
    original_device = None
    
    try:
        if TORCH_AVAILABLE and torch.cuda.is_available():
            if device and device.startswith('cuda:'):
                original_device = torch.cuda.current_device()
                torch.cuda.set_device(int(device.split(':')[1]))
            yield device or f"cuda:{torch.cuda.current_device()}"
        elif CUPY_AVAILABLE and cp.cuda.is_available():
            yield "cupy"
        else:
            yield None
    finally:
        if TORCH_AVAILABLE and torch.cuda.is_available() and original_device is not None:
            torch.cuda.set_device(original_device)


def cuda_tensor_to_numpy(tensor) -> Optional[np.ndarray]:
    """Convert CUDA tensor to NumPy array."""
    if not NUMPY_AVAILABLE:
        return None
    
    try:
        if TORCH_AVAILABLE and hasattr(tensor, 'cpu'):
            return tensor.cpu().numpy()
        elif CUPY_AVAILABLE and hasattr(tensor, 'get'):
            return cp.asnumpy(tensor)
        elif hasattr(tensor, 'numpy'):
            return tensor.numpy()
        else:
            return np.array(tensor)
    except Exception as e:
        logger.info(f"Warning: Failed to convert tensor to numpy: {e}")
        return None


def numpy_to_cuda_tensor(array: np.ndarray, device: Optional[str] = None) -> Any:
    """Convert NumPy array to CUDA tensor."""
    if not NUMPY_AVAILABLE or not cuda_available():
        return array
    
    try:
        if TORCH_AVAILABLE and torch.cuda.is_available():
            tensor = torch.from_numpy(array)
            if device and device.startswith('cuda:'):
                return tensor.to(device)
            else:
                return tensor.cuda()
        elif CUPY_AVAILABLE and cp.cuda.is_available():
            return cp.asarray(array)
        else:
            return array
    except Exception as e:
        logger.info(f"Warning: Failed to convert numpy to CUDA tensor: {e}")
        return array


def cuda_batch_operation(operation_func, data_list: List[Any], batch_size: int = 1000) -> List[Any]:
    """Perform CUDA operation on data in batches."""
    if not cuda_available() or len(data_list) < batch_size:
        return [operation_func(item) for item in data_list]
    
    results = []
    for i in range(0, len(data_list), batch_size):
        batch = data_list[i:i + batch_size]
        try:
            batch_results = operation_func(batch)
            if isinstance(batch_results, list):
                results.extend(batch_results)
            else:
                results.append(batch_results)
        except Exception as e:
            logger.info(f"Warning: CUDA batch operation failed, falling back to CPU: {e}")
            # Fallback to CPU
            for item in batch:
                results.append(operation_func(item))
    
    return results


def cuda_benchmark(operation_func, test_data: List[Any], num_iterations: int = 10) -> Dict[str, float]:
    """Benchmark CUDA operation performance."""
    if not cuda_available():
        return {'cuda_available': False}
    
    # Warm up
    try:
        operation_func(test_data[:min(10, len(test_data))])
        cuda_synchronize()
    except Exception:
        return {'cuda_available': False}
    
    # Benchmark
    times = []
    for _ in range(num_iterations):
        start_time = time.time()
        try:
            operation_func(test_data)
            cuda_synchronize()
            end_time = time.time()
            times.append(end_time - start_time)
        except Exception as e:
            logger.info(f"Warning: CUDA benchmark failed: {e}")
            return {'cuda_available': False}
    
    if not times:
        return {'cuda_available': False}
    
    return {
        'cuda_available': True,
        'avg_time': sum(times) / len(times),
        'min_time': min(times),
        'max_time': max(times),
        'std_time': (sum((t - sum(times) / len(times)) ** 2 for t in times) / len(times)) ** 0.5,
        'throughput': len(test_data) / (sum(times) / len(times)),
    }


def cuda_memory_usage() -> Dict[str, int]:
    """Get current CUDA memory usage."""
    if not cuda_available():
        return {'available': False}
    
    try:
        if TORCH_AVAILABLE and torch.cuda.is_available():
            device_id = torch.cuda.current_device()
            return {
                'available': True,
                'allocated': torch.cuda.memory_allocated(device_id),
                'reserved': torch.cuda.memory_reserved(device_id),
                'max_allocated': torch.cuda.max_memory_allocated(device_id),
                'max_reserved': torch.cuda.max_memory_reserved(device_id),
            }
        elif CUPY_AVAILABLE and cp.cuda.is_available():
            mempool = cp.get_default_memory_pool()
            return {
                'available': True,
                'allocated': mempool.used_bytes(),
                'reserved': mempool.total_bytes(),
                'max_allocated': mempool.used_bytes(),
                'max_reserved': mempool.total_bytes(),
            }
    except Exception as e:
        logger.info(f"Warning: Failed to get CUDA memory usage: {e}")
    
    return {'available': False}


def cuda_device_info() -> Dict[str, Any]:
    """Get detailed CUDA device information."""
    if not cuda_available():
        return {'available': False}
    
    try:
        if TORCH_AVAILABLE and torch.cuda.is_available():
            device_id = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(device_id)
            return {
                'available': True,
                'device_id': device_id,
                'name': props.name,
                'compute_capability': props.major,
                'minor_compute_capability': props.minor,
                'total_memory': props.total_memory,
                'multiprocessor_count': props.multi_processor_count,
                'max_threads_per_block': props.max_threads_per_block,
                'max_threads_per_multiprocessor': props.max_threads_per_multiprocessor,
                'warp_size': props.warp_size,
                'memory_clock_rate': props.memory_clock_rate,
                'memory_bus_width': props.memory_bus_width,
            }
        elif CUPY_AVAILABLE and cp.cuda.is_available():
            return {
                'available': True,
                'device_id': 0,
                'name': 'CuPy Device',
                'compute_capability': None,
                'total_memory': cp.cuda.runtime.memGetInfo()[1],
            }
    except Exception as e:
        logger.info(f"Warning: Failed to get CUDA device info: {e}")
    
    return {'available': False}
