"""
Hardware acceleration module for DubChain.

This module provides comprehensive hardware acceleration support including:
- Multi-platform GPU acceleration (CUDA, OpenCL, Metal)
- CPU SIMD optimizations (AVX-512, ARM NEON)
- Hardware capability detection and automatic selection
- Performance benchmarking and comparison
"""

from .detection import (
    HardwareDetector,
    HardwareCapabilities,
    AccelerationType,
    PlatformType,
)
from .base import (
    HardwareAccelerator,
    AccelerationConfig,
    PerformanceMetrics,
)
from .cuda import (
    CUDAAccelerator,
    CUDAConfig,
)
from .opencl import (
    OpenCLAccelerator,
    OpenCLConfig,
)
from .metal import (
    MetalAccelerator,
    MetalConfig,
)
from .cpu import (
    CPUAccelerator,
    CPUConfig,
    SIMDType,
)
from .manager import (
    HardwareManager,
    HardwareManagerConfig,
)

__all__ = [
    # Detection
    "HardwareDetector",
    "HardwareCapabilities", 
    "AccelerationType",
    "PlatformType",
    # Base
    "HardwareAccelerator",
    "AccelerationConfig",
    "PerformanceMetrics",
    # CUDA
    "CUDAAccelerator",
    "CUDAConfig",
    # OpenCL
    "OpenCLAccelerator", 
    "OpenCLConfig",
    # Metal
    "MetalAccelerator",
    "MetalConfig",
    # CPU
    "CPUAccelerator",
    "CPUConfig",
    "SIMDType",
    # Manager
    "HardwareManager",
    "HardwareManagerConfig",
]
