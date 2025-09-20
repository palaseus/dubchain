"""
Optimized Memory Management implementation for DubChain.

This module provides performance optimizations for memory management including:
- Allocation reduction and buffer reuse
- Garbage collection tuning
- Memory pooling and object reuse
- Zero-copy operations
- Memory-mapped files
"""

import gc
import mmap
import os
import threading
import time
import weakref
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Callable, BinaryIO
import sys

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

from ..performance.optimizations import OptimizationManager, OptimizationFallback


@dataclass
class BufferPool:
    """Pool of reusable buffers."""
    buffers: deque = field(default_factory=deque)
    buffer_size: int = 0
    max_pool_size: int = 100
    total_allocated: int = 0
    total_reused: int = 0


@dataclass
class MemoryStats:
    """Memory usage statistics."""
    total_allocated: int = 0
    total_freed: int = 0
    peak_usage: int = 0
    current_usage: int = 0
    gc_collections: int = 0
    buffer_reuses: int = 0


@dataclass
class MemoryConfig:
    """Memory optimization configuration."""
    enable_buffer_reuse: bool = True
    enable_gc_tuning: bool = True
    enable_memory_pooling: bool = True
    enable_zero_copy: bool = True
    buffer_pool_size: int = 1000
    gc_threshold_multiplier: float = 0.8
    memory_pressure_threshold: float = 0.8
    auto_cleanup_interval: float = 60.0  # seconds


class OptimizedMemory:
    """
    Optimized Memory Management with performance enhancements.
    
    Features:
    - Allocation reduction and buffer reuse
    - Garbage collection tuning
    - Memory pooling and object reuse
    - Zero-copy operations
    - Memory-mapped files
    """
    
    def __init__(self, optimization_manager: OptimizationManager, config: Optional[MemoryConfig] = None):
        """Initialize optimized memory manager."""
        self.optimization_manager = optimization_manager
        self.config = config or MemoryConfig()
        
        # Buffer pools by size
        self.buffer_pools: Dict[int, BufferPool] = {}
        self.pool_lock = threading.RLock()
        
        # Object pools for common types
        self.object_pools: Dict[type, deque] = defaultdict(deque)
        self.object_pool_lock = threading.RLock()
        
        # Memory-mapped files
        self.mmap_files: Dict[str, mmap.mmap] = {}
        self.mmap_lock = threading.RLock()
        
        # Memory statistics
        self.stats = MemoryStats()
        self.stats_lock = threading.RLock()
        
        # GC tuning
        self.original_gc_thresholds = None
        self.gc_tuning_enabled = False
        
        # Memory pressure monitoring
        self.memory_pressure_threshold = self.config.memory_pressure_threshold
        self.last_cleanup_time = time.time()
        
        # Performance metrics
        self.metrics = {
            "total_allocations": 0,
            "total_deallocations": 0,
            "buffer_reuses": 0,
            "object_reuses": 0,
            "gc_collections": 0,
            "memory_pressure_events": 0,
            "avg_allocation_time": 0.0,
        }
        
        # Thread safety
        self._metrics_lock = threading.Lock()
        
        # Initialize optimizations
        self._initialize_optimizations()
    
    def _initialize_optimizations(self):
        """Initialize memory optimizations."""
        if self.config.enable_gc_tuning:
            self._enable_gc_tuning()
        
        if self.config.enable_buffer_reuse:
            self._initialize_buffer_pools()
    
    def _enable_gc_tuning(self):
        """Enable garbage collection tuning."""
        if not self.optimization_manager.is_optimization_enabled("memory_gc_tuning"):
            return
        
        # Store original thresholds
        self.original_gc_thresholds = gc.get_threshold()
        
        # Optimize GC thresholds
        new_thresholds = (
            int(self.original_gc_thresholds[0] * self.config.gc_threshold_multiplier),
            int(self.original_gc_thresholds[1] * self.config.gc_threshold_multiplier),
            int(self.original_gc_thresholds[2] * self.config.gc_threshold_multiplier),
        )
        
        gc.set_threshold(*new_thresholds)
        self.gc_tuning_enabled = True
    
    def _disable_gc_tuning(self):
        """Disable garbage collection tuning."""
        if self.gc_tuning_enabled and self.original_gc_thresholds:
            gc.set_threshold(*self.original_gc_thresholds)
            self.gc_tuning_enabled = False
    
    def _initialize_buffer_pools(self):
        """Initialize buffer pools for common sizes."""
        common_sizes = [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
        
        for size in common_sizes:
            self.buffer_pools[size] = BufferPool(
                buffer_size=size,
                max_pool_size=self.config.buffer_pool_size // len(common_sizes)
            )
    
    @OptimizationFallback
    def get_reusable_buffer(self, size: int) -> bytearray:
        """
        Get a reusable buffer of specified size.
        
        Args:
            size: Buffer size in bytes
            
        Returns:
            Reusable buffer
        """
        if not self.optimization_manager.is_optimization_enabled("memory_buffer_reuse"):
            self.metrics["total_allocations"] += 1
            return bytearray(size)
        
        with self.pool_lock:
            # Find appropriate pool
            pool_size = self._find_pool_size(size)
            
            if pool_size in self.buffer_pools:
                pool = self.buffer_pools[pool_size]
                
                if pool.buffers:
                    # Reuse existing buffer
                    buffer = pool.buffers.popleft()
                    buffer[:] = b'\x00' * size  # Clear buffer
                    pool.total_reused += 1
                    self.metrics["buffer_reuses"] += 1
                    return buffer
                else:
                    # Allocate new buffer
                    buffer = bytearray(pool_size)
                    pool.total_allocated += 1
                    self.metrics["total_allocations"] += 1
                    return buffer
            else:
                # No pool available, allocate directly
                self.metrics["total_allocations"] += 1
                return bytearray(size)
    
    def _find_pool_size(self, requested_size: int) -> int:
        """Find appropriate pool size for requested buffer size."""
        # Find smallest pool size that can accommodate the request
        for pool_size in sorted(self.buffer_pools.keys()):
            if pool_size >= requested_size:
                return pool_size
        
        # If no pool is large enough, return the requested size
        return requested_size
    
    @OptimizationFallback
    def return_buffer(self, buffer: bytearray):
        """
        Return a buffer to the pool for reuse.
        
        Args:
            buffer: Buffer to return
        """
        if not self.optimization_manager.is_optimization_enabled("memory_buffer_reuse"):
            return
        
        buffer_size = len(buffer)
        
        with self.pool_lock:
            pool_size = self._find_pool_size(buffer_size)
            
            if pool_size in self.buffer_pools:
                pool = self.buffer_pools[pool_size]
                
                if len(pool.buffers) < pool.max_pool_size:
                    pool.buffers.append(buffer)
                    self.metrics["total_deallocations"] += 1
    
    @OptimizationFallback
    def get_reusable_object(self, obj_type: type, *args, **kwargs) -> Any:
        """
        Get a reusable object from the pool.
        
        Args:
            obj_type: Type of object to get
            *args: Arguments for object creation
            **kwargs: Keyword arguments for object creation
            
        Returns:
            Reusable object
        """
        if not self.optimization_manager.is_optimization_enabled("memory_object_pooling"):
            return obj_type(*args, **kwargs)
        
        with self.object_pool_lock:
            if obj_type in self.object_pools and self.object_pools[obj_type]:
                # Reuse existing object
                obj = self.object_pools[obj_type].popleft()
                self.metrics["object_reuses"] += 1
                
                # Reset object state if it has a reset method
                if hasattr(obj, 'reset'):
                    obj.reset(*args, **kwargs)
                elif hasattr(obj, '__init__'):
                    obj.__init__(*args, **kwargs)
                
                return obj
            else:
                # Create new object
                self.metrics["total_allocations"] += 1
                return obj_type(*args, **kwargs)
    
    @OptimizationFallback
    def return_object(self, obj: Any):
        """
        Return an object to the pool for reuse.
        
        Args:
            obj: Object to return
        """
        if not self.optimization_manager.is_optimization_enabled("memory_object_pooling"):
            return
        
        obj_type = type(obj)
        
        with self.object_pool_lock:
            if len(self.object_pools[obj_type]) < self.config.buffer_pool_size:
                self.object_pools[obj_type].append(obj)
                self.metrics["total_deallocations"] += 1
    
    @OptimizationFallback
    def create_memory_mapped_file(self, file_path: str, size: int) -> mmap.mmap:
        """
        Create a memory-mapped file for zero-copy operations.
        
        Args:
            file_path: Path to the file
            size: Size of the file
            
        Returns:
            Memory-mapped file object
        """
        if not self.optimization_manager.is_optimization_enabled("memory_zero_copy"):
            # Fallback to regular file
            with open(file_path, 'wb') as f:
                f.write(b'\x00' * size)
            return None
        
        try:
            with self.mmap_lock:
                if file_path in self.mmap_files:
                    return self.mmap_files[file_path]
                
                # Create file if it doesn't exist
                if not os.path.exists(file_path):
                    with open(file_path, 'wb') as f:
                        f.write(b'\x00' * size)
                
                # Open and map the file
                with open(file_path, 'r+b') as f:
                    mmap_file = mmap.mmap(f.fileno(), size)
                    self.mmap_files[file_path] = mmap_file
                    return mmap_file
        except Exception as e:
            print(f"Failed to create memory-mapped file: {e}")
            return None
    
    def close_memory_mapped_file(self, file_path: str):
        """Close a memory-mapped file."""
        with self.mmap_lock:
            if file_path in self.mmap_files:
                self.mmap_files[file_path].close()
                del self.mmap_files[file_path]
    
    def optimize_gc_settings(self):
        """Optimize garbage collection settings."""
        if not self.optimization_manager.is_optimization_enabled("memory_gc_tuning"):
            return
        
        # Enable GC tuning if not already enabled
        if not self.gc_tuning_enabled:
            self._enable_gc_tuning()
        
        # Force garbage collection
        collected = gc.collect()
        self.metrics["gc_collections"] += 1
        
        # Update statistics
        with self.stats_lock:
            self.stats.gc_collections += 1
        
        return collected
    
    def restore_gc_settings(self):
        """Restore original garbage collection settings."""
        self._disable_gc_tuning()
    
    def check_memory_pressure(self) -> float:
        """Check current memory pressure."""
        if not PSUTIL_AVAILABLE:
            return 0.0
        
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            memory_percent = process.memory_percent()
            
            with self.stats_lock:
                self.stats.current_usage = memory_info.rss
                if memory_info.rss > self.stats.peak_usage:
                    self.stats.peak_usage = memory_info.rss
            
            return memory_percent / 100.0
        except Exception:
            return 0.0
    
    def handle_memory_pressure(self):
        """Handle memory pressure by cleaning up resources."""
        memory_pressure = self.check_memory_pressure()
        
        if memory_pressure > self.memory_pressure_threshold:
            self.metrics["memory_pressure_events"] += 1
            
            # Clean up buffer pools
            self._cleanup_buffer_pools()
            
            # Clean up object pools
            self._cleanup_object_pools()
            
            # Force garbage collection
            self.optimize_gc_settings()
            
            # Close unused memory-mapped files
            self._cleanup_mmap_files()
    
    def _cleanup_buffer_pools(self):
        """Clean up buffer pools to reduce memory usage."""
        with self.pool_lock:
            for pool in self.buffer_pools.values():
                # Keep only half of the buffers
                target_size = pool.max_pool_size // 2
                while len(pool.buffers) > target_size:
                    pool.buffers.popleft()
    
    def _cleanup_object_pools(self):
        """Clean up object pools to reduce memory usage."""
        with self.object_pool_lock:
            for obj_type, pool in self.object_pools.items():
                # Keep only half of the objects
                target_size = self.config.buffer_pool_size // 2
                while len(pool) > target_size:
                    pool.popleft()
    
    def _cleanup_mmap_files(self):
        """Clean up unused memory-mapped files."""
        with self.mmap_lock:
            # Close files that haven't been accessed recently
            current_time = time.time()
            files_to_close = []
            
            for file_path, mmap_file in self.mmap_files.items():
                # Simple cleanup - close files older than 1 hour
                if current_time - self.last_cleanup_time > 3600:
                    files_to_close.append(file_path)
            
            for file_path in files_to_close:
                self.close_memory_mapped_file(file_path)
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics."""
        with self.stats_lock:
            stats = {
                "total_allocated": self.stats.total_allocated,
                "total_freed": self.stats.total_freed,
                "peak_usage": self.stats.peak_usage,
                "current_usage": self.stats.current_usage,
                "gc_collections": self.stats.gc_collections,
                "buffer_reuses": self.stats.buffer_reuses,
            }
        
        # Add buffer pool statistics
        buffer_pool_stats = {}
        with self.pool_lock:
            for size, pool in self.buffer_pools.items():
                buffer_pool_stats[f"pool_{size}"] = {
                    "available_buffers": len(pool.buffers),
                    "total_allocated": pool.total_allocated,
                    "total_reused": pool.total_reused,
                }
        
        # Add object pool statistics
        object_pool_stats = {}
        with self.object_pool_lock:
            for obj_type, pool in self.object_pools.items():
                object_pool_stats[obj_type.__name__] = {
                    "available_objects": len(pool),
                }
        
        return {
            **stats,
            "buffer_pools": buffer_pool_stats,
            "object_pools": object_pool_stats,
            "memory_mapped_files": len(self.mmap_files),
            "gc_tuning_enabled": self.gc_tuning_enabled,
            "memory_pressure": self.check_memory_pressure(),
        }
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics."""
        return {
            **self.metrics,
            "memory_stats": self.get_memory_stats(),
            "optimization_enabled": {
                "buffer_reuse": self.optimization_manager.is_optimization_enabled("memory_buffer_reuse"),
                "gc_tuning": self.optimization_manager.is_optimization_enabled("memory_gc_tuning"),
                "object_pooling": self.optimization_manager.is_optimization_enabled("memory_object_pooling"),
                "zero_copy": self.optimization_manager.is_optimization_enabled("memory_zero_copy"),
            }
        }
    
    def cleanup(self):
        """Clean up all resources."""
        # Close all memory-mapped files
        with self.mmap_lock:
            for file_path in list(self.mmap_files.keys()):
                self.close_memory_mapped_file(file_path)
        
        # Clear buffer pools
        with self.pool_lock:
            for pool in self.buffer_pools.values():
                pool.buffers.clear()
        
        # Clear object pools
        with self.object_pool_lock:
            for pool in self.object_pools.values():
                pool.clear()
        
        # Restore GC settings
        self.restore_gc_settings()
    
    def __del__(self):
        """Cleanup on destruction."""
        self.cleanup()
