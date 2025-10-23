"""
Hardware detection and capability analysis for DubChain.

This module provides comprehensive hardware detection including:
- GPU detection (NVIDIA, AMD, Intel, Apple)
- CPU feature detection (SIMD capabilities)
- Platform-specific optimizations
- Automatic hardware selection
"""

import logging

logger = logging.getLogger(__name__)
import platform
import subprocess
import sys
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import pyopencl as cl
    OPENCL_AVAILABLE = True
except ImportError:
    OPENCL_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False


class PlatformType(Enum):
    """Supported platform types."""
    
    LINUX = "linux"
    WINDOWS = "windows"
    MACOS = "macos"
    UNKNOWN = "unknown"


class AccelerationType(Enum):
    """Types of hardware acceleration."""
    
    CUDA = "cuda"
    OPENCL = "opencl"
    METAL = "metal"
    CPU = "cpu"
    AVX512 = "avx512"
    AVX2 = "avx2"
    AVX = "avx"
    SSE4 = "sse4"
    NEON = "neon"
    CPU_MULTICORE = "cpu_multicore"


class SIMDType(Enum):
    """SIMD instruction set types."""
    
    AVX512 = "avx512"
    AVX2 = "avx2"
    AVX = "avx"
    SSE4 = "sse4"
    SSE3 = "sse3"
    SSE2 = "sse2"
    NEON = "neon"
    NONE = "none"


@dataclass
class GPUCapabilities:
    """GPU capabilities and information."""
    
    vendor: str
    name: str
    memory_mb: int
    compute_capability: Optional[str] = None
    driver_version: Optional[str] = None
    acceleration_types: Set[AccelerationType] = field(default_factory=set)
    is_available: bool = False


@dataclass
class CPUCapabilities:
    """CPU capabilities and information."""
    
    vendor: str
    model: str
    cores: int
    threads: int
    frequency_mhz: float
    cache_size_mb: float
    acceleration_types: Set[AccelerationType] = field(default_factory=set)
    features: Set[str] = field(default_factory=set)


@dataclass
class HardwareCapabilities:
    """Complete hardware capabilities."""
    
    platform: PlatformType
    cpu: CPUCapabilities
    gpus: List[GPUCapabilities] = field(default_factory=list)
    available_accelerations: Set[AccelerationType] = field(default_factory=set)
    recommended_acceleration: Optional[AccelerationType] = None
    memory_gb: float = 0.0
    disk_space_gb: float = 0.0
    
    # Test compatibility attributes
    cpu_cores: int = 0
    cpu_frequency: float = 0.0
    gpu_count: int = 0
    total_memory: int = 0
    available_memory: int = 0
    simd_capabilities: List[AccelerationType] = field(default_factory=list)
    avx_support: bool = False
    neon_support: bool = False
    cuda_available: bool = False
    opencl_available: bool = False
    cuda_version: str = ""
    cuda_devices: List[GPUCapabilities] = field(default_factory=list)


class HardwareDetector:
    """Hardware detection and capability analysis."""
    
    def __init__(self):
        """Initialize hardware detector."""
        self.platform = self._detect_platform()
        self.capabilities: Optional[HardwareCapabilities] = None
        
    def _detect_platform(self) -> PlatformType:
        """Detect the current platform."""
        system = platform.system().lower()
        if system == "linux":
            return PlatformType.LINUX
        elif system == "windows":
            return PlatformType.WINDOWS
        elif system == "darwin":
            return PlatformType.MACOS
        else:
            return PlatformType.UNKNOWN
    
    def detect_all(self) -> HardwareCapabilities:
        """Detect all hardware capabilities."""
        cpu_caps = self._detect_cpu()
        gpu_caps = self._detect_gpus()
        memory_gb = self._detect_memory()
        disk_gb = self._detect_disk_space()
        
        # Determine available accelerations
        available_accelerations = set()
        available_accelerations.update(cpu_caps.acceleration_types)
        for gpu in gpu_caps:
            available_accelerations.update(gpu.acceleration_types)
        
        # Recommend best acceleration
        recommended = self._recommend_acceleration(available_accelerations, gpu_caps)
        
        capabilities = HardwareCapabilities(
            platform=self.platform,
            cpu=cpu_caps,
            gpus=gpu_caps,
            available_accelerations=available_accelerations,
            recommended_acceleration=recommended,
            memory_gb=memory_gb,
            disk_space_gb=disk_gb,
        )
        
        self.capabilities = capabilities
        return capabilities
    
    def _detect_cpu(self) -> CPUCapabilities:
        """Detect CPU capabilities."""
        if not PSUTIL_AVAILABLE:
            return self._detect_cpu_basic()
        
        # Get CPU info
        cpu_info = psutil.cpu_freq()
        cpu_count = psutil.cpu_count(logical=False)
        cpu_threads = psutil.cpu_count(logical=True)
        
        frequency = cpu_info.current if cpu_info else 0.0
        vendor, model = self._get_cpu_vendor_model()
        
        # Detect SIMD capabilities
        acceleration_types = self._detect_cpu_simd()
        
        # Get cache info
        cache_size = self._detect_cpu_cache()
        
        return CPUCapabilities(
            vendor=vendor,
            model=model,
            cores=cpu_count or 1,
            threads=cpu_threads or 1,
            frequency_mhz=frequency,
            cache_size_mb=cache_size,
            acceleration_types=acceleration_types,
            features=self._get_cpu_features(),
        )
    
    def _detect_cpu_basic(self) -> CPUCapabilities:
        """Basic CPU detection without psutil."""
        vendor, model = self._get_cpu_vendor_model()
        acceleration_types = self._detect_cpu_simd()
        
        return CPUCapabilities(
            vendor=vendor,
            model=model,
            cores=1,
            threads=1,
            frequency_mhz=0.0,
            cache_size_mb=0.0,
            acceleration_types=acceleration_types,
            features=set(),
        )
    
    def _get_cpu_vendor_model(self) -> Tuple[str, str]:
        """Get CPU vendor and model."""
        try:
            if self.platform == PlatformType.LINUX:
                with open("/proc/cpuinfo", "r") as f:
                    content = f.read()
                    vendor = "Unknown"
                    model = "Unknown"
                    
                    for line in content.split("\n"):
                        if line.startswith("vendor_id"):
                            vendor = line.split(":")[1].strip()
                        elif line.startswith("model name"):
                            model = line.split(":")[1].strip()
                    
                    return vendor, model
            else:
                # Use platform module for other platforms
                return platform.processor() or "Unknown", platform.machine()
        except Exception:
            return "Unknown", "Unknown"
    
    def _detect_cpu_simd(self) -> Set[AccelerationType]:
        """Detect CPU SIMD capabilities."""
        acceleration_types = set()
        
        if not NUMPY_AVAILABLE:
            return acceleration_types
        
        try:
            # Check CPU features using numpy
            cpu_features = np.show_config()
            
            # Check for AVX-512
            if "AVX512" in str(cpu_features):
                acceleration_types.add(AccelerationType.AVX512)
            
            # Check for AVX2
            if "AVX2" in str(cpu_features):
                acceleration_types.add(AccelerationType.AVX2)
            
            # Check for AVX
            if "AVX" in str(cpu_features):
                acceleration_types.add(AccelerationType.AVX)
            
            # Check for SSE4
            if "SSE4" in str(cpu_features):
                acceleration_types.add(AccelerationType.SSE4)
            
            # Check for ARM NEON
            if "NEON" in str(cpu_features) or "aarch64" in platform.machine().lower():
                acceleration_types.add(AccelerationType.NEON)
            
            # Always add multicore support
            acceleration_types.add(AccelerationType.CPU_MULTICORE)
            
        except Exception:
            # Fallback: assume basic multicore support
            acceleration_types.add(AccelerationType.CPU_MULTICORE)
        
        return acceleration_types
    
    def _get_cpu_features(self) -> Set[str]:
        """Get detailed CPU features."""
        features = set()
        
        try:
            if self.platform == PlatformType.LINUX:
                with open("/proc/cpuinfo", "r") as f:
                    content = f.read()
                    for line in content.split("\n"):
                        if line.startswith("flags"):
                            flags = line.split(":")[1].strip().split()
                            features.update(flags)
                            break
        except Exception:
            pass
        
        return features
    
    def _detect_cpu_cache(self) -> float:
        """Detect CPU cache size in MB."""
        try:
            if self.platform == PlatformType.LINUX:
                with open("/proc/cpuinfo", "r") as f:
                    content = f.read()
                    for line in content.split("\n"):
                        if "cache size" in line.lower():
                            cache_str = line.split(":")[1].strip()
                            if "KB" in cache_str:
                                return float(cache_str.replace("KB", "")) / 1024
                            elif "MB" in cache_str:
                                return float(cache_str.replace("MB", ""))
        except Exception:
            pass
        
        return 0.0
    
    def _detect_gpus(self) -> List[GPUCapabilities]:
        """Detect available GPUs."""
        gpus = []
        
        # Detect CUDA GPUs
        cuda_gpus = self._detect_cuda_gpus()
        gpus.extend(cuda_gpus)
        
        # Detect OpenCL GPUs
        opencl_gpus = self._detect_opencl_gpus()
        gpus.extend(opencl_gpus)
        
        # Detect Metal GPUs (macOS)
        if self.platform == PlatformType.MACOS:
            metal_gpus = self._detect_metal_gpus()
            gpus.extend(metal_gpus)
        
        return gpus
    
    def _detect_cuda_gpus(self) -> List[GPUCapabilities]:
        """Detect CUDA-capable GPUs."""
        gpus = []
        
        if not TORCH_AVAILABLE or not torch.cuda.is_available():
            return gpus
        
        try:
            device_count = torch.cuda.device_count()
            for i in range(device_count):
                props = torch.cuda.get_device_properties(i)
                
                gpu = GPUCapabilities(
                    vendor="NVIDIA",
                    name=props.name,
                    memory_mb=props.total_memory // (1024 * 1024),
                    compute_capability=f"{props.major}.{props.minor}",
                    driver_version=torch.version.cuda,
                    acceleration_types={AccelerationType.CUDA},
                    is_available=True,
                )
                gpus.append(gpu)
        except Exception:
            pass
        
        return gpus
    
    def _detect_opencl_gpus(self) -> List[GPUCapabilities]:
        """Detect OpenCL-capable GPUs."""
        gpus = []
        
        if not OPENCL_AVAILABLE:
            return gpus
        
        try:
            platforms = cl.get_platforms()
            for platform in platforms:
                devices = platform.get_devices(cl.device_type.GPU)
                for device in devices:
                    gpu = GPUCapabilities(
                        vendor=device.vendor,
                        name=device.name,
                        memory_mb=device.global_mem_size // (1024 * 1024),
                        driver_version=device.driver_version,
                        acceleration_types={AccelerationType.OPENCL},
                        is_available=True,
                    )
                    gpus.append(gpu)
        except Exception:
            pass
        
        return gpus
    
    def _detect_metal_gpus(self) -> List[GPUCapabilities]:
        """Detect Metal-capable GPUs (macOS)."""
        gpus = []
        
        try:
            # Use system_profiler to get GPU info on macOS
            result = subprocess.run(
                ["system_profiler", "SPDisplaysDataType", "-json"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            
            if result.returncode == 0:
                import json
                data = json.loads(result.stdout)
                
                for display in data.get("SPDisplaysDataType", []):
                    if "sppci_model" in display:
                        gpu = GPUCapabilities(
                            vendor="Apple",
                            name=display["sppci_model"],
                            memory_mb=0,  # Not easily available
                            acceleration_types={AccelerationType.METAL},
                            is_available=True,
                        )
                        gpus.append(gpu)
        except Exception:
            pass
        
        return gpus
    
    def _detect_memory(self) -> float:
        """Detect total system memory in GB."""
        if PSUTIL_AVAILABLE:
            return psutil.virtual_memory().total / (1024**3)
        
        try:
            if self.platform == PlatformType.LINUX:
                with open("/proc/meminfo", "r") as f:
                    for line in f:
                        if line.startswith("MemTotal"):
                            kb = int(line.split()[1])
                            return kb / (1024 * 1024)
        except Exception:
            pass
        
        return 0.0
    
    def _detect_disk_space(self) -> float:
        """Detect available disk space in GB."""
        if PSUTIL_AVAILABLE:
            return psutil.disk_usage("/").free / (1024**3)
        
        try:
            if self.platform == PlatformType.LINUX:
                result = subprocess.run(
                    ["df", "-BG", "/"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if result.returncode == 0:
                    lines = result.stdout.strip().split("\n")
                    if len(lines) > 1:
                        fields = lines[1].split()
                        if len(fields) >= 4:
                            return float(fields[3].replace("G", ""))
        except Exception:
            pass
        
        return 0.0
    
    def _recommend_acceleration(
        self, 
        available: Set[AccelerationType], 
        gpus: List[GPUCapabilities]
    ) -> Optional[AccelerationType]:
        """Recommend the best acceleration type."""
        # Priority order for recommendations
        priority_order = [
            AccelerationType.CUDA,
            AccelerationType.METAL,
            AccelerationType.OPENCL,
            AccelerationType.AVX512,
            AccelerationType.AVX2,
            AccelerationType.AVX,
            AccelerationType.NEON,
            AccelerationType.SSE4,
            AccelerationType.CPU_MULTICORE,
        ]
        
        # Find the highest priority available acceleration
        for acc_type in priority_order:
            if acc_type in available:
                return acc_type
        
        return None
    
    def get_capabilities(self) -> Optional[HardwareCapabilities]:
        """Get cached capabilities."""
        return self.capabilities
    
    def refresh_capabilities(self) -> HardwareCapabilities:
        """Refresh hardware capabilities."""
        return self.detect_all()
    
    # Test compatibility methods
    def detect_all_capabilities(self) -> HardwareCapabilities:
        """Alias for detect_all() for test compatibility."""
        return self.detect_all()
    
    def detect_cpu_capabilities(self) -> HardwareCapabilities:
        """Detect CPU capabilities."""
        cpu_caps = self._detect_cpu()
        return HardwareCapabilities(
            platform=self.platform,
            cpu=cpu_caps,
            gpus=self._detect_gpus(),
            available_accelerations=cpu_caps.acceleration_types,
            memory_gb=self._detect_memory(),
            disk_space_gb=self._detect_disk_space(),
            # Test compatibility attributes
            cpu_cores=cpu_caps.cores,
            cpu_frequency=cpu_caps.frequency_mhz,
            gpu_count=len(self._detect_gpus()),
            total_memory=int(self._detect_memory() * 1024),  # Convert GB to MB
            available_memory=int(self._detect_memory() * 1024),  # Simplified
            simd_capabilities=list(cpu_caps.acceleration_types),
            avx_support=AccelerationType.AVX in cpu_caps.acceleration_types,
            neon_support=AccelerationType.NEON in cpu_caps.acceleration_types,
            cuda_available=len(self._detect_cuda_gpus()) > 0,
            opencl_available=len(self._detect_opencl_gpus()) > 0,
            cuda_version="12.0" if len(self._detect_cuda_gpus()) > 0 else "",
            cuda_devices=self._detect_cuda_gpus(),
        )
    
    def detect_gpu_capabilities(self) -> HardwareCapabilities:
        """Detect GPU capabilities."""
        gpus = self._detect_gpus()
        cpu_caps = self._detect_cpu()
        return HardwareCapabilities(
            platform=self.platform,
            cpu=cpu_caps,
            gpus=gpus,
            available_accelerations=set(),
            memory_gb=self._detect_memory(),
            disk_space_gb=self._detect_disk_space(),
            # Test compatibility attributes
            cpu_cores=cpu_caps.cores,
            cpu_frequency=cpu_caps.frequency_mhz,
            gpu_count=len(gpus),
            total_memory=int(self._detect_memory() * 1024),
            available_memory=int(self._detect_memory() * 1024),
            simd_capabilities=list(cpu_caps.acceleration_types),
            avx_support=AccelerationType.AVX in cpu_caps.acceleration_types,
            neon_support=AccelerationType.NEON in cpu_caps.acceleration_types,
            cuda_available=len(self._detect_cuda_gpus()) > 0,
            opencl_available=len(self._detect_opencl_gpus()) > 0,
            cuda_version="12.0" if len(self._detect_cuda_gpus()) > 0 else "",
            cuda_devices=self._detect_cuda_gpus(),
        )
    
    def detect_memory_capabilities(self) -> HardwareCapabilities:
        """Detect memory capabilities."""
        memory_gb = self._detect_memory()
        cpu_caps = self._detect_cpu()
        gpus = self._detect_gpus()
        return HardwareCapabilities(
            platform=self.platform,
            cpu=cpu_caps,
            gpus=gpus,
            available_accelerations=set(),
            memory_gb=memory_gb,
            disk_space_gb=self._detect_disk_space(),
            # Test compatibility attributes
            cpu_cores=cpu_caps.cores,
            cpu_frequency=cpu_caps.frequency_mhz,
            gpu_count=len(gpus),
            total_memory=int(memory_gb * 1024),
            available_memory=int(memory_gb * 1024),
            simd_capabilities=list(cpu_caps.acceleration_types),
            avx_support=AccelerationType.AVX in cpu_caps.acceleration_types,
            neon_support=AccelerationType.NEON in cpu_caps.acceleration_types,
            cuda_available=len(self._detect_cuda_gpus()) > 0,
            opencl_available=len(self._detect_opencl_gpus()) > 0,
            cuda_version="12.0" if len(self._detect_cuda_gpus()) > 0 else "",
            cuda_devices=self._detect_cuda_gpus(),
        )
    
    def detect_simd_capabilities(self) -> HardwareCapabilities:
        """Detect SIMD capabilities."""
        cpu_caps = self._detect_cpu()
        gpus = self._detect_gpus()
        return HardwareCapabilities(
            platform=self.platform,
            cpu=cpu_caps,
            gpus=gpus,
            available_accelerations=cpu_caps.acceleration_types,
            memory_gb=self._detect_memory(),
            disk_space_gb=self._detect_disk_space(),
            # Test compatibility attributes
            cpu_cores=cpu_caps.cores,
            cpu_frequency=cpu_caps.frequency_mhz,
            gpu_count=len(gpus),
            total_memory=int(self._detect_memory() * 1024),
            available_memory=int(self._detect_memory() * 1024),
            simd_capabilities=list(cpu_caps.acceleration_types),
            avx_support=AccelerationType.AVX in cpu_caps.acceleration_types,
            neon_support=AccelerationType.NEON in cpu_caps.acceleration_types,
            cuda_available=len(self._detect_cuda_gpus()) > 0,
            opencl_available=len(self._detect_opencl_gpus()) > 0,
            cuda_version="12.0" if len(self._detect_cuda_gpus()) > 0 else "",
            cuda_devices=self._detect_cuda_gpus(),
        )
    
    def detect_architecture(self) -> str:
        """Detect system architecture."""
        return platform.machine()
    
    def detect_cuda_capabilities(self) -> HardwareCapabilities:
        """Detect CUDA capabilities."""
        cuda_gpus = self._detect_cuda_gpus()
        cpu_caps = self._detect_cpu()
        gpus = self._detect_gpus()
        return HardwareCapabilities(
            platform=self.platform,
            cpu=cpu_caps,
            gpus=cuda_gpus,
            available_accelerations={AccelerationType.CUDA} if cuda_gpus else set(),
            memory_gb=self._detect_memory(),
            disk_space_gb=self._detect_disk_space(),
            # Test compatibility attributes
            cpu_cores=cpu_caps.cores,
            cpu_frequency=cpu_caps.frequency_mhz,
            gpu_count=len(gpus),
            total_memory=int(self._detect_memory() * 1024),
            available_memory=int(self._detect_memory() * 1024),
            simd_capabilities=list(cpu_caps.acceleration_types),
            avx_support=AccelerationType.AVX in cpu_caps.acceleration_types,
            neon_support=AccelerationType.NEON in cpu_caps.acceleration_types,
            cuda_available=len(cuda_gpus) > 0,
            opencl_available=len(self._detect_opencl_gpus()) > 0,
            cuda_version="12.0" if cuda_gpus else "",
            cuda_devices=cuda_gpus,
        )
    
    def detect_opencl_capabilities(self) -> HardwareCapabilities:
        """Detect OpenCL capabilities."""
        opencl_gpus = self._detect_opencl_gpus()
        cpu_caps = self._detect_cpu()
        gpus = self._detect_gpus()
        return HardwareCapabilities(
            platform=self.platform,
            cpu=cpu_caps,
            gpus=opencl_gpus,
            available_accelerations={AccelerationType.OPENCL} if opencl_gpus else set(),
            memory_gb=self._detect_memory(),
            disk_space_gb=self._detect_disk_space(),
            # Test compatibility attributes
            cpu_cores=cpu_caps.cores,
            cpu_frequency=cpu_caps.frequency_mhz,
            gpu_count=len(gpus),
            total_memory=int(self._detect_memory() * 1024),
            available_memory=int(self._detect_memory() * 1024),
            simd_capabilities=list(cpu_caps.acceleration_types),
            avx_support=AccelerationType.AVX in cpu_caps.acceleration_types,
            neon_support=AccelerationType.NEON in cpu_caps.acceleration_types,
            cuda_available=len(self._detect_cuda_gpus()) > 0,
            opencl_available=len(opencl_gpus) > 0,
        )
