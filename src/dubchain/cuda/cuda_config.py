"""
CUDA Configuration for DubChain.

This module provides configuration management for CUDA operations throughout
the entire codebase, ensuring consistent GPU acceleration settings.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
import os


@dataclass
class CUDAConfig:
    """Comprehensive CUDA configuration for DubChain."""
    
    # Core CUDA settings
    enable_cuda: bool = True
    device_id: int = 0
    fallback_to_cpu: bool = True
    
    # Memory management
    memory_limit_mb: int = 1024
    memory_fraction: float = 0.8
    enable_memory_pool: bool = True
    
    # Performance settings
    min_batch_size_gpu: int = 100
    max_batch_size: int = 10000
    chunk_size: int = 1000
    
    # Algorithm-specific settings
    enable_crypto_gpu: bool = True
    enable_consensus_gpu: bool = True
    enable_sharding_gpu: bool = True
    enable_networking_gpu: bool = True
    enable_storage_gpu: bool = True
    
    # Testing settings
    enable_test_gpu: bool = True
    test_gpu_fallback: bool = True
    benchmark_gpu: bool = True
    
    # Debugging settings
    enable_cuda_logging: bool = False
    log_memory_usage: bool = False
    profile_gpu_operations: bool = False
    
    # Environment overrides
    environment_overrides: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Apply environment variable overrides."""
        self._apply_environment_overrides()
    
    def _apply_environment_overrides(self):
        """Apply environment variable overrides."""
        env_mappings = {
            'DUBCHAIN_CUDA_ENABLE': ('enable_cuda', bool),
            'DUBCHAIN_CUDA_DEVICE': ('device_id', int),
            'DUBCHAIN_CUDA_MEMORY_LIMIT': ('memory_limit_mb', int),
            'DUBCHAIN_CUDA_MIN_BATCH_SIZE': ('min_batch_size_gpu', int),
            'DUBCHAIN_CUDA_MAX_BATCH_SIZE': ('max_batch_size', int),
            'DUBCHAIN_CUDA_CHUNK_SIZE': ('chunk_size', int),
            'DUBCHAIN_CUDA_CRYPTO': ('enable_crypto_gpu', bool),
            'DUBCHAIN_CUDA_CONSENSUS': ('enable_consensus_gpu', bool),
            'DUBCHAIN_CUDA_SHARDING': ('enable_sharding_gpu', bool),
            'DUBCHAIN_CUDA_NETWORKING': ('enable_networking_gpu', bool),
            'DUBCHAIN_CUDA_STORAGE': ('enable_storage_gpu', bool),
            'DUBCHAIN_CUDA_TEST': ('enable_test_gpu', bool),
            'DUBCHAIN_CUDA_LOGGING': ('enable_cuda_logging', bool),
            'DUBCHAIN_CUDA_PROFILE': ('profile_gpu_operations', bool),
        }
        
        for env_var, (attr_name, attr_type) in env_mappings.items():
            env_value = os.getenv(env_var)
            if env_value is not None:
                try:
                    if attr_type == bool:
                        value = env_value.lower() in ('true', '1', 'yes', 'on')
                    else:
                        value = attr_type(env_value)
                    setattr(self, attr_name, value)
                    self.environment_overrides[env_var] = value
                except (ValueError, TypeError) as e:
                    print(f"Warning: Invalid environment variable {env_var}={env_value}: {e}")
    
    def get_algorithm_config(self, algorithm: str) -> Dict[str, Any]:
        """Get configuration for a specific algorithm."""
        base_config = {
            'enable_cuda': self.enable_cuda,
            'device_id': self.device_id,
            'fallback_to_cpu': self.fallback_to_cpu,
            'memory_limit_mb': self.memory_limit_mb,
            'min_batch_size_gpu': self.min_batch_size_gpu,
            'max_batch_size': self.max_batch_size,
            'chunk_size': self.chunk_size,
        }
        
        # Algorithm-specific settings
        if algorithm == 'crypto':
            base_config.update({
                'enable_gpu': self.enable_crypto_gpu,
            })
        elif algorithm == 'consensus':
            base_config.update({
                'enable_gpu': self.enable_consensus_gpu,
            })
        elif algorithm == 'sharding':
            base_config.update({
                'enable_gpu': self.enable_sharding_gpu,
            })
        elif algorithm == 'networking':
            base_config.update({
                'enable_gpu': self.enable_networking_gpu,
            })
        elif algorithm == 'storage':
            base_config.update({
                'enable_gpu': self.enable_storage_gpu,
            })
        elif algorithm == 'testing':
            base_config.update({
                'enable_gpu': self.enable_test_gpu,
                'fallback_to_cpu': self.test_gpu_fallback,
            })
        
        return base_config
    
    def should_use_gpu(self, algorithm: str, batch_size: int = 1) -> bool:
        """Determine if GPU should be used for a given algorithm and batch size."""
        if not self.enable_cuda:
            return False
        
        # Check algorithm-specific enablement
        if algorithm == 'crypto' and not self.enable_crypto_gpu:
            return False
        elif algorithm == 'consensus' and not self.enable_consensus_gpu:
            return False
        elif algorithm == 'sharding' and not self.enable_sharding_gpu:
            return False
        elif algorithm == 'networking' and not self.enable_networking_gpu:
            return False
        elif algorithm == 'storage' and not self.enable_storage_gpu:
            return False
        elif algorithm == 'testing' and not self.enable_test_gpu:
            return False
        
        # Check batch size threshold
        if batch_size < self.min_batch_size_gpu:
            return False
        
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'enable_cuda': self.enable_cuda,
            'device_id': self.device_id,
            'fallback_to_cpu': self.fallback_to_cpu,
            'memory_limit_mb': self.memory_limit_mb,
            'memory_fraction': self.memory_fraction,
            'enable_memory_pool': self.enable_memory_pool,
            'min_batch_size_gpu': self.min_batch_size_gpu,
            'max_batch_size': self.max_batch_size,
            'chunk_size': self.chunk_size,
            'enable_crypto_gpu': self.enable_crypto_gpu,
            'enable_consensus_gpu': self.enable_consensus_gpu,
            'enable_sharding_gpu': self.enable_sharding_gpu,
            'enable_networking_gpu': self.enable_networking_gpu,
            'enable_storage_gpu': self.enable_storage_gpu,
            'enable_test_gpu': self.enable_test_gpu,
            'test_gpu_fallback': self.test_gpu_fallback,
            'benchmark_gpu': self.benchmark_gpu,
            'enable_cuda_logging': self.enable_cuda_logging,
            'log_memory_usage': self.log_memory_usage,
            'profile_gpu_operations': self.profile_gpu_operations,
            'environment_overrides': self.environment_overrides,
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'CUDAConfig':
        """Create configuration from dictionary."""
        return cls(**config_dict)
    
    def __str__(self) -> str:
        """String representation of configuration."""
        return f"CUDAConfig(enable_cuda={self.enable_cuda}, device_id={self.device_id}, " \
               f"memory_limit_mb={self.memory_limit_mb}, min_batch_size_gpu={self.min_batch_size_gpu})"


# Global CUDA configuration instance
_global_cuda_config: Optional[CUDAConfig] = None


def get_global_cuda_config() -> CUDAConfig:
    """Get the global CUDA configuration."""
    global _global_cuda_config
    if _global_cuda_config is None:
        _global_cuda_config = CUDAConfig()
    return _global_cuda_config


def set_global_cuda_config(config: CUDAConfig) -> None:
    """Set the global CUDA configuration."""
    global _global_cuda_config
    _global_cuda_config = config


def reset_global_cuda_config() -> None:
    """Reset the global CUDA configuration to defaults."""
    global _global_cuda_config
    _global_cuda_config = CUDAConfig()
