"""
Advanced Memory Management System for DubChain.

This module implements comprehensive memory management including:
- Memory pools for frequent allocations
- Memory usage monitoring and limits
- Garbage collection optimization
- Memory leak detection and prevention
- Object lifecycle management
- Memory profiling and analytics
"""

import gc
import os
import sys
import threading
import time
import tracemalloc
import weakref
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import psutil


class MemoryPoolType(Enum):
    """Types of memory pools."""

    BLOCK_POOL = "block_pool"
    TRANSACTION_POOL = "transaction_pool"
    UTXO_POOL = "utxo_pool"
    HASH_POOL = "hash_pool"
    SIGNATURE_POOL = "signature_pool"
    GENERAL_POOL = "general_pool"


class MemoryEventType(Enum):
    """Types of memory events."""

    ALLOCATION = "allocation"
    DEALLOCATION = "deallocation"
    POOL_HIT = "pool_hit"
    POOL_MISS = "pool_miss"
    GC_TRIGGERED = "gc_triggered"
    LEAK_DETECTED = "leak_detected"
    LIMIT_EXCEEDED = "limit_exceeded"


@dataclass
class MemoryStats:
    """Memory usage statistics."""

    total_memory: int = 0
    used_memory: int = 0
    free_memory: int = 0
    memory_percentage: float = 0.0
    process_memory: int = 0
    peak_memory: int = 0
    gc_count: int = 0
    gc_time: float = 0.0
    pool_hits: int = 0
    pool_misses: int = 0
    allocations: int = 0
    deallocations: int = 0
    timestamp: float = field(default_factory=time.time)


@dataclass
class MemoryEvent:
    """Memory event record."""

    event_type: MemoryEventType
    pool_type: Optional[MemoryPoolType]
    size: int
    timestamp: float = field(default_factory=time.time)
    stack_trace: Optional[List[str]] = None
    object_type: Optional[str] = None
    pool_size: int = 0
    total_memory: int = 0


@dataclass
class MemoryLeak:
    """Memory leak detection result."""

    object_type: str
    count: int
    total_size: int
    first_seen: float
    last_seen: float
    growth_rate: float
    severity: str  # low, medium, high, critical
    stack_traces: List[List[str]] = field(default_factory=list)


class MemoryPool:
    """Memory pool for efficient object allocation."""

    def __init__(
        self,
        pool_type: MemoryPoolType,
        max_size: int = 1000,
        object_factory: Optional[Callable] = None,
        cleanup_callback: Optional[Callable] = None,
    ):
        self.pool_type = pool_type
        self.max_size = max_size
        self.object_factory = object_factory
        self.cleanup_callback = cleanup_callback

        self._pool: deque = deque(maxlen=max_size)
        self._allocated_objects: Set[Any] = set()
        self._stats = {
            "hits": 0,
            "misses": 0,
            "allocations": 0,
            "deallocations": 0,
            "total_size": 0,
        }

        self._lock = threading.RLock()

    def get_object(self, *args, **kwargs) -> Any:
        """Get an object from the pool or create a new one."""
        with self._lock:
            if self._pool:
                # Pool hit
                obj = self._pool.popleft()
                self._stats["hits"] += 1
                self._allocated_objects.add(obj)
                return obj
            else:
                # Pool miss - create new object
                if self.object_factory:
                    obj = self.object_factory(*args, **kwargs)
                else:
                    obj = object()

                self._stats["misses"] += 1
                self._stats["allocations"] += 1
                self._allocated_objects.add(obj)
                return obj

    def return_object(self, obj: Any) -> None:
        """Return an object to the pool."""
        with self._lock:
            if obj in self._allocated_objects:
                self._allocated_objects.remove(obj)

                # Clean up object if callback provided
                if self.cleanup_callback:
                    self.cleanup_callback(obj)

                # Add to pool if there's space
                if len(self._pool) < self.max_size:
                    self._pool.append(obj)
                    self._stats["deallocations"] += 1
                else:
                    # Pool is full, object will be garbage collected
                    pass

    def clear(self) -> None:
        """Clear the pool."""
        with self._lock:
            self._pool.clear()
            self._allocated_objects.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics."""
        with self._lock:
            hit_rate = 0.0
            total_requests = self._stats["hits"] + self._stats["misses"]
            if total_requests > 0:
                hit_rate = self._stats["hits"] / total_requests

            return {
                "pool_type": self.pool_type.value,
                "pool_size": len(self._pool),
                "max_size": self.max_size,
                "allocated_objects": len(self._allocated_objects),
                "hits": self._stats["hits"],
                "misses": self._stats["misses"],
                "hit_rate": hit_rate,
                "allocations": self._stats["allocations"],
                "deallocations": self._stats["deallocations"],
            }


class MemoryMonitor:
    """Memory usage monitor."""

    def __init__(self, update_interval: float = 1.0):
        self.update_interval = update_interval
        self._running = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._stats_history: deque = deque(maxlen=1000)
        self._current_stats: Optional[MemoryStats] = None
        self._peak_memory = 0
        self._lock = threading.RLock()

        # Memory tracking
        self._object_counts: Dict[str, int] = defaultdict(int)
        self._object_sizes: Dict[str, int] = defaultdict(int)
        self._allocation_times: Dict[str, float] = {}

        # Callbacks
        self._memory_limit_callback: Optional[Callable[[MemoryStats], None]] = None
        self._leak_detection_callback: Optional[Callable[[MemoryLeak], None]] = None

    def start(self) -> None:
        """Start memory monitoring."""
        with self._lock:
            if self._running:
                return

            self._running = True
            self._monitor_thread = threading.Thread(
                target=self._monitor_worker, daemon=True
            )
            self._monitor_thread.start()

    def stop(self) -> None:
        """Stop memory monitoring."""
        with self._lock:
            self._running = False
            if self._monitor_thread:
                self._monitor_thread.join(timeout=5.0)

    def get_current_stats(self) -> MemoryStats:
        """Get current memory statistics."""
        with self._lock:
            if self._current_stats:
                return self._current_stats
            else:
                return self._collect_stats()

    def get_stats_history(self) -> List[MemoryStats]:
        """Get memory statistics history."""
        with self._lock:
            return list(self._stats_history)

    def set_memory_limit_callback(
        self, callback: Callable[[MemoryStats], None]
    ) -> None:
        """Set callback for memory limit exceeded."""
        self._memory_limit_callback = callback

    def set_leak_detection_callback(
        self, callback: Callable[[MemoryLeak], None]
    ) -> None:
        """Set callback for memory leak detection."""
        self._leak_detection_callback = callback

    def track_allocation(self, obj: Any, size: int) -> None:
        """Track object allocation."""
        obj_type = type(obj).__name__

        with self._lock:
            self._object_counts[obj_type] += 1
            self._object_sizes[obj_type] += size
            self._allocation_times[id(obj)] = time.time()

    def track_deallocation(self, obj: Any) -> None:
        """Track object deallocation."""
        obj_type = type(obj).__name__
        obj_id = id(obj)

        with self._lock:
            if obj_type in self._object_counts:
                self._object_counts[obj_type] -= 1
                if self._object_counts[obj_type] <= 0:
                    del self._object_counts[obj_type]

            if obj_type in self._object_sizes:
                # Estimate size (this is approximate)
                estimated_size = self._object_sizes[obj_type] // max(
                    1, self._object_counts.get(obj_type, 1)
                )
                self._object_sizes[obj_type] -= estimated_size
                if self._object_sizes[obj_type] <= 0:
                    del self._object_sizes[obj_type]

            if obj_id in self._allocation_times:
                del self._allocation_times[obj_id]

    def _monitor_worker(self) -> None:
        """Background worker for memory monitoring."""
        while self._running:
            try:
                stats = self._collect_stats()

                with self._lock:
                    self._current_stats = stats
                    self._stats_history.append(stats)

                    # Update peak memory
                    if stats.process_memory > self._peak_memory:
                        self._peak_memory = stats.process_memory
                        stats.peak_memory = self._peak_memory

                # Check for memory limits
                if self._memory_limit_callback and stats.memory_percentage > 90.0:
                    self._memory_limit_callback(stats)

                # Detect memory leaks
                self._detect_memory_leaks()

                time.sleep(self.update_interval)

            except Exception as e:
                print(f"Memory monitoring error: {e}")
                time.sleep(self.update_interval)

    def _collect_stats(self) -> MemoryStats:
        """Collect current memory statistics."""
        # System memory
        memory = psutil.virtual_memory()
        process = psutil.Process()
        process_memory = process.memory_info().rss

        # Garbage collection stats
        gc_stats = gc.get_stats()
        gc_count = sum(stat["collections"] for stat in gc_stats)
        gc_time = sum(stat["collected"] for stat in gc_stats)

        return MemoryStats(
            total_memory=memory.total,
            used_memory=memory.used,
            free_memory=memory.available,
            memory_percentage=memory.percent,
            process_memory=process_memory,
            peak_memory=self._peak_memory,
            gc_count=gc_count,
            gc_time=gc_time,
            pool_hits=0,  # Will be updated by memory manager
            pool_misses=0,  # Will be updated by memory manager
            allocations=sum(self._object_counts.values()),
            deallocations=0,  # Will be updated by memory manager
        )

    def _detect_memory_leaks(self) -> None:
        """Detect potential memory leaks."""
        if not self._leak_detection_callback:
            return

        current_time = time.time()
        leaks = []

        # Check for objects that have been allocated for too long
        for obj_id, alloc_time in self._allocation_times.items():
            age = current_time - alloc_time
            if age > 3600:  # 1 hour
                # Potential leak
                obj_type = "unknown"
                for obj_type_name in self._object_counts:
                    if self._object_counts[obj_type_name] > 0:
                        obj_type = obj_type_name
                        break

                leak = MemoryLeak(
                    object_type=obj_type,
                    count=self._object_counts.get(obj_type, 0),
                    total_size=self._object_sizes.get(obj_type, 0),
                    first_seen=alloc_time,
                    last_seen=current_time,
                    growth_rate=0.0,  # Would need historical data
                    severity="medium" if age > 7200 else "low",
                )
                leaks.append(leak)

        # Report leaks
        for leak in leaks:
            self._leak_detection_callback(leak)


class GarbageCollectionManager:
    """Advanced garbage collection management."""

    def __init__(self, auto_gc: bool = True, gc_threshold: float = 0.8):
        self.auto_gc = auto_gc
        self.gc_threshold = gc_threshold
        self._gc_stats = {
            "manual_gc_count": 0,
            "auto_gc_count": 0,
            "total_gc_time": 0.0,
            "last_gc_time": 0.0,
        }
        self._lock = threading.RLock()

        # GC optimization
        self._generation_thresholds = [700, 10, 10]  # Default thresholds
        self._gc_callbacks: List[Callable] = []

    def set_generation_thresholds(self, gen0: int, gen1: int, gen2: int) -> None:
        """Set garbage collection generation thresholds."""
        with self._lock:
            self._generation_thresholds = [gen0, gen1, gen2]
            gc.set_threshold(gen0, gen1, gen2)

    def add_gc_callback(self, callback: Callable) -> None:
        """Add callback to be called before garbage collection."""
        with self._lock:
            self._gc_callbacks.append(callback)

    def force_gc(self, generation: int = 2) -> Dict[str, Any]:
        """Force garbage collection."""
        with self._lock:
            start_time = time.time()

            # Call pre-GC callbacks
            for callback in self._gc_callbacks:
                try:
                    callback()
                except Exception as e:
                    print(f"GC callback error: {e}")

            # Perform garbage collection
            collected = gc.collect(generation)

            end_time = time.time()
            gc_time = end_time - start_time

            # Update stats
            self._gc_stats["manual_gc_count"] += 1
            self._gc_stats["total_gc_time"] += gc_time
            self._gc_stats["last_gc_time"] = gc_time

            return {
                "collected": collected,
                "gc_time": gc_time,
                "generation": generation,
                "stats": self._gc_stats.copy(),
            }

    def auto_gc_check(self, memory_percentage: float) -> bool:
        """Check if automatic garbage collection should be triggered."""
        if not self.auto_gc:
            return False

        if memory_percentage > self.gc_threshold * 100:
            self.force_gc()
            self._gc_stats["auto_gc_count"] += 1
            return True

        return False

    def get_gc_stats(self) -> Dict[str, Any]:
        """Get garbage collection statistics."""
        with self._lock:
            return self._gc_stats.copy()

    def optimize_gc(self) -> None:
        """Optimize garbage collection settings."""
        with self._lock:
            # Adjust thresholds based on usage patterns
            # This is a simplified optimization
            current_stats = gc.get_stats()

            # If we're doing too much GC, increase thresholds
            if current_stats[0]["collections"] > 100:
                new_thresholds = [t * 2 for t in self._generation_thresholds]
                self.set_generation_thresholds(*new_thresholds)


class MemoryProfiler:
    """Memory profiler for detailed analysis."""

    def __init__(self, enable_tracing: bool = True):
        self.enable_tracing = enable_tracing
        self._tracing_enabled = False
        self._snapshots: List[tracemalloc.Snapshot] = []
        self._peak_memory = 0
        self._allocation_traces: Dict[str, List[tracemalloc.Traceback]] = {}

        if self.enable_tracing:
            self.start_tracing()

    def start_tracing(self) -> None:
        """Start memory tracing."""
        if not self._tracing_enabled:
            tracemalloc.start()
            self._tracing_enabled = True

    def stop_tracing(self) -> None:
        """Stop memory tracing."""
        if self._tracing_enabled:
            tracemalloc.stop()
            self._tracing_enabled = False

    def take_snapshot(self) -> tracemalloc.Snapshot:
        """Take a memory snapshot."""
        if self._tracing_enabled:
            snapshot = tracemalloc.take_snapshot()
            self._snapshots.append(snapshot)
            return snapshot
        else:
            raise RuntimeError("Memory tracing not enabled")

    def compare_snapshots(
        self, snapshot1: tracemalloc.Snapshot, snapshot2: tracemalloc.Snapshot
    ) -> List[tracemalloc.Statistic]:
        """Compare two memory snapshots."""
        top_stats = snapshot2.compare_to(snapshot1, "lineno")
        return top_stats

    def get_memory_usage_by_traceback(self) -> Dict[str, tracemalloc.Statistic]:
        """Get memory usage grouped by traceback."""
        if not self._snapshots:
            return {}

        current_snapshot = self._snapshots[-1]
        stats_by_traceback = {}

        for stat in current_snapshot.statistics("traceback"):
            traceback_key = str(stat.traceback)
            stats_by_traceback[traceback_key] = stat

        return stats_by_traceback

    def get_top_memory_allocations(
        self, limit: int = 10
    ) -> List[tracemalloc.Statistic]:
        """Get top memory allocations."""
        if not self._snapshots:
            return []

        current_snapshot = self._snapshots[-1]
        return current_snapshot.statistics("lineno")[:limit]

    def detect_memory_leaks(self) -> List[Dict[str, Any]]:
        """Detect potential memory leaks by comparing snapshots."""
        if len(self._snapshots) < 2:
            return []

        # Compare last two snapshots
        old_snapshot = self._snapshots[-2]
        new_snapshot = self._snapshots[-1]

        diff_stats = self.compare_snapshots(old_snapshot, new_snapshot)

        leaks = []
        for stat in diff_stats:
            if stat.size_diff > 1024 * 1024:  # 1MB threshold
                leak_info = {
                    "size_diff": stat.size_diff,
                    "count_diff": stat.count_diff,
                    "traceback": str(stat.traceback),
                    "filename": stat.traceback.format()[0]
                    if stat.traceback.format()
                    else "unknown",
                }
                leaks.append(leak_info)

        return leaks


class MemoryManager:
    """Main memory management system."""

    def __init__(
        self,
        max_memory_mb: int = 1024,
        enable_monitoring: bool = True,
        enable_profiling: bool = False,
        auto_gc: bool = True,
    ):
        self.max_memory_mb = max_memory_mb
        self.max_memory_bytes = max_memory_mb * 1024 * 1024

        # Initialize components
        self._pools: Dict[MemoryPoolType, MemoryPool] = {}
        self._monitor = MemoryMonitor() if enable_monitoring else None
        self._gc_manager = GarbageCollectionManager(auto_gc=auto_gc)
        self._profiler = (
            MemoryProfiler(enable_tracing=enable_profiling)
            if enable_profiling
            else None
        )

        # Event tracking
        self._events: deque = deque(maxlen=10000)
        self._event_callbacks: List[Callable[[MemoryEvent], None]] = []

        # Memory limits
        self._memory_limits: Dict[str, int] = {}
        self._limit_callbacks: Dict[str, Callable] = {}

        # Object lifecycle tracking
        self._object_registry: Dict[int, Dict[str, Any]] = {}
        self._weak_refs: Dict[int, weakref.ref] = {}

        self._lock = threading.RLock()

        # Start monitoring
        if self._monitor:
            self._monitor.start()
            self._monitor.set_memory_limit_callback(self._on_memory_limit_exceeded)
            self._monitor.set_leak_detection_callback(self._on_leak_detected)

    def create_pool(
        self,
        pool_type: MemoryPoolType,
        max_size: int = 1000,
        object_factory: Optional[Callable] = None,
        cleanup_callback: Optional[Callable] = None,
    ) -> MemoryPool:
        """Create a memory pool."""
        with self._lock:
            pool = MemoryPool(
                pool_type=pool_type,
                max_size=max_size,
                object_factory=object_factory,
                cleanup_callback=cleanup_callback,
            )
            self._pools[pool_type] = pool
            return pool

    def get_pool(self, pool_type: MemoryPoolType) -> Optional[MemoryPool]:
        """Get a memory pool."""
        return self._pools.get(pool_type)

    def allocate_object(
        self,
        obj: Any,
        pool_type: Optional[MemoryPoolType] = None,
        size: Optional[int] = None,
    ) -> Any:
        """Allocate an object through memory management."""
        with self._lock:
            # Try to get from pool first
            if pool_type and pool_type in self._pools:
                pool = self._pools[pool_type]
                pooled_obj = pool.get_object()
                if pooled_obj is not None:
                    self._record_event(
                        MemoryEvent(
                            event_type=MemoryEventType.POOL_HIT,
                            pool_type=pool_type,
                            size=size or 0,
                            object_type=type(pooled_obj).__name__,
                        )
                    )

                    # Register pooled object for lifecycle tracking
                    obj_id = id(pooled_obj)
                    self._object_registry[obj_id] = {
                        "object": pooled_obj,
                        "type": type(pooled_obj).__name__,
                        "created": time.time(),
                        "pool_type": pool_type,
                        "size": size or 0,
                    }

                    # Create weak reference for cleanup tracking (only for objects that support it)
                    try:

                        def cleanup_callback(ref):
                            self._on_object_cleanup(obj_id)

                        self._weak_refs[obj_id] = weakref.ref(
                            pooled_obj, cleanup_callback
                        )
                    except TypeError:
                        # Some objects (like strings, numbers) cannot have weak references
                        # Just track them without weak reference cleanup
                        pass

                    # Track in monitor
                    if self._monitor:
                        self._monitor.track_allocation(pooled_obj, size or 0)

                    return pooled_obj

            # Pool miss or no pool
            if pool_type:
                self._record_event(
                    MemoryEvent(
                        event_type=MemoryEventType.POOL_MISS,
                        pool_type=pool_type,
                        size=size or 0,
                        object_type=type(obj).__name__,
                    )
                )

            # Track allocation
            self._record_event(
                MemoryEvent(
                    event_type=MemoryEventType.ALLOCATION,
                    pool_type=pool_type,
                    size=size or 0,
                    object_type=type(obj).__name__,
                )
            )

            # Register object for lifecycle tracking
            obj_id = id(obj)
            self._object_registry[obj_id] = {
                "object": obj,
                "type": type(obj).__name__,
                "created": time.time(),
                "pool_type": pool_type,
                "size": size or 0,
            }

            # Create weak reference for cleanup tracking (only for objects that support it)
            try:

                def cleanup_callback(ref):
                    self._on_object_cleanup(obj_id)

                self._weak_refs[obj_id] = weakref.ref(obj, cleanup_callback)
            except TypeError:
                # Some objects (like strings, numbers) cannot have weak references
                # Just track them without weak reference cleanup
                pass

            # Track in monitor
            if self._monitor:
                self._monitor.track_allocation(obj, size or 0)

            return obj

    def deallocate_object(
        self, obj: Any, pool_type: Optional[MemoryPoolType] = None
    ) -> None:
        """Deallocate an object through memory management."""
        with self._lock:
            obj_id = id(obj)

            # Return to pool if available
            if pool_type and pool_type in self._pools:
                pool = self._pools[pool_type]
                pool.return_object(obj)

            # Record deallocation event
            self._record_event(
                MemoryEvent(
                    event_type=MemoryEventType.DEALLOCATION,
                    pool_type=pool_type,
                    size=self._object_registry.get(obj_id, {}).get("size", 0),
                    object_type=type(obj).__name__,
                )
            )

            # Remove from registry
            if obj_id in self._object_registry:
                del self._object_registry[obj_id]

            if obj_id in self._weak_refs:
                del self._weak_refs[obj_id]

            # Track in monitor
            if self._monitor:
                self._monitor.track_deallocation(obj)

    def set_memory_limit(
        self, limit_name: str, limit_mb: int, callback: Optional[Callable] = None
    ) -> None:
        """Set a memory limit with callback."""
        with self._lock:
            self._memory_limits[limit_name] = limit_mb * 1024 * 1024
            if callback:
                self._limit_callbacks[limit_name] = callback

    def check_memory_limits(self) -> Dict[str, bool]:
        """Check if memory limits are exceeded."""
        if not self._monitor:
            return {}

        current_stats = self._monitor.get_current_stats()
        results = {}

        for limit_name, limit_bytes in self._memory_limits.items():
            exceeded = current_stats.process_memory > limit_bytes
            results[limit_name] = exceeded

            if exceeded and limit_name in self._limit_callbacks:
                try:
                    self._limit_callbacks[limit_name]()
                except Exception as e:
                    print(f"Memory limit callback error: {e}")

        return results

    def force_garbage_collection(self, generation: int = 2) -> Dict[str, Any]:
        """Force garbage collection."""
        result = self._gc_manager.force_gc(generation)

        self._record_event(
            MemoryEvent(
                event_type=MemoryEventType.GC_TRIGGERED,
                pool_type=None,
                size=0,
                object_type="garbage_collection",
            )
        )

        return result

    def get_memory_stats(self) -> MemoryStats:
        """Get comprehensive memory statistics."""
        if self._monitor:
            stats = self._monitor.get_current_stats()
        else:
            stats = MemoryStats()

        # Add pool statistics
        total_pool_hits = 0
        total_pool_misses = 0

        for pool in self._pools.values():
            pool_stats = pool.get_stats()
            total_pool_hits += pool_stats["hits"]
            total_pool_misses += pool_stats["misses"]

        stats.pool_hits = total_pool_hits
        stats.pool_misses = total_pool_misses

        return stats

    def get_pool_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all memory pools."""
        return {
            pool_type.value: pool.get_stats() for pool_type, pool in self._pools.items()
        }

    def add_event_callback(self, callback: Callable[[MemoryEvent], None]) -> None:
        """Add callback for memory events."""
        self._event_callbacks.append(callback)

    def get_memory_events(self, limit: int = 100) -> List[MemoryEvent]:
        """Get recent memory events."""
        return list(self._events)[-limit:]

    def cleanup(self) -> None:
        """Clean up memory management resources."""
        with self._lock:
            # Stop monitoring
            if self._monitor:
                self._monitor.stop()

            # Clear pools
            for pool in self._pools.values():
                pool.clear()
            self._pools.clear()

            # Clear registries
            self._object_registry.clear()
            self._weak_refs.clear()

            # Stop profiling
            if self._profiler:
                self._profiler.stop_tracing()

    def _record_event(self, event: MemoryEvent) -> None:
        """Record a memory event."""
        self._events.append(event)

        # Call event callbacks
        for callback in self._event_callbacks:
            try:
                callback(event)
            except Exception as e:
                print(f"Memory event callback error: {e}")

    def _on_memory_limit_exceeded(self, stats: MemoryStats) -> None:
        """Handle memory limit exceeded."""
        self._record_event(
            MemoryEvent(
                event_type=MemoryEventType.LIMIT_EXCEEDED,
                pool_type=None,
                size=stats.process_memory,
                object_type="system",
            )
        )

        # Trigger garbage collection
        self._gc_manager.auto_gc_check(stats.memory_percentage)

    def _on_leak_detected(self, leak: MemoryLeak) -> None:
        """Handle memory leak detection."""
        self._record_event(
            MemoryEvent(
                event_type=MemoryEventType.LEAK_DETECTED,
                pool_type=None,
                size=leak.total_size,
                object_type=leak.object_type,
            )
        )

    def _on_object_cleanup(self, obj_id: int) -> None:
        """Handle object cleanup via weak reference."""
        with self._lock:
            if obj_id in self._object_registry:
                obj_info = self._object_registry[obj_id]
                del self._object_registry[obj_id]

            if obj_id in self._weak_refs:
                del self._weak_refs[obj_id]

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()
        # Clear pools to ensure cleanup
        self._pools.clear()
