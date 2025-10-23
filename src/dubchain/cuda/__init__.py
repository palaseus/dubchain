"""
CUDA Integration Module for DubChain.

This module provides comprehensive CUDA support throughout the entire codebase,
including automatic GPU acceleration for computationally intensive operations.
"""

import logging

logger = logging.getLogger(__name__)
from .cuda_manager import CUDAManager, get_global_cuda_manager
from .cuda_utils import (
    cuda_available,
    get_cuda_device,
    get_cuda_memory_info,
    cuda_synchronize,
    cuda_memory_cleanup,
)
from .cuda_config import CUDAConfig

__all__ = [
    'CUDAManager',
    'get_global_cuda_manager',
    'cuda_available',
    'get_cuda_device',
    'get_cuda_memory_info',
    'cuda_synchronize',
    'cuda_memory_cleanup',
    'CUDAConfig',
]
