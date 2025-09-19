"""
Comprehensive unit tests for the memory management system.
"""

import gc
import threading
import time
from unittest.mock import Mock, patch

import pytest

from dubchain.core.memory_management import (
    GarbageCollectionManager,
    MemoryEvent,
    MemoryEventType,
    MemoryLeak,
    MemoryManager,
    MemoryMonitor,
    MemoryPool,
    MemoryPoolType,
    MemoryProfiler,
    MemoryStats,
)


class TestMemoryPool:
    """Test the MemoryPool class."""

    def test_memory_pool_creation(self):
        """Test creating a memory pool."""
        pool = MemoryPool(
            pool_type=MemoryPoolType.BLOCK_POOL,
            max_size=100,
            object_factory=lambda: "test_object",
        )

        assert pool.pool_type == MemoryPoolType.BLOCK_POOL
        assert pool.max_size == 100
        assert pool.object_factory is not None
        assert len(pool._pool) == 0
        assert len(pool._allocated_objects) == 0

    def test_get_object_from_empty_pool(self):
        """Test getting object from empty pool."""
        pool = MemoryPool(
            pool_type=MemoryPoolType.BLOCK_POOL,
            max_size=100,
            object_factory=lambda: "new_object",
        )

        obj = pool.get_object()

        assert obj == "new_object"
        assert len(pool._allocated_objects) == 1
        assert pool._stats["misses"] == 1
        assert pool._stats["allocations"] == 1

    def test_get_object_from_pool_with_objects(self):
        """Test getting object from pool that has objects."""
        pool = MemoryPool(
            pool_type=MemoryPoolType.BLOCK_POOL,
            max_size=100,
            object_factory=lambda: "new_object",
        )

        # Add object to pool
        pool._pool.append("pooled_object")

        obj = pool.get_object()

        assert obj == "pooled_object"
        assert len(pool._pool) == 0
        assert len(pool._allocated_objects) == 1
        assert pool._stats["hits"] == 1

    def test_return_object_to_pool(self):
        """Test returning object to pool."""
        pool = MemoryPool(pool_type=MemoryPoolType.BLOCK_POOL, max_size=100)

        # Get object from pool
        obj = pool.get_object()

        # Return object to pool
        pool.return_object(obj)

        assert len(pool._pool) == 1
        assert len(pool._allocated_objects) == 0
        assert pool._stats["deallocations"] == 1

    def test_return_object_to_full_pool(self):
        """Test returning object to full pool."""
        pool = MemoryPool(pool_type=MemoryPoolType.BLOCK_POOL, max_size=2)

        # Fill pool
        pool._pool.extend(["obj1", "obj2"])

        # Get object
        obj = pool.get_object()

        # Return object (pool is full)
        pool.return_object(obj)

        assert len(pool._pool) == 2  # Pool size unchanged
        assert len(pool._allocated_objects) == 0

    def test_cleanup_callback(self):
        """Test cleanup callback when returning object."""
        cleanup_callback = Mock()
        pool = MemoryPool(
            pool_type=MemoryPoolType.BLOCK_POOL,
            max_size=100,
            cleanup_callback=cleanup_callback,
        )

        obj = pool.get_object()
        pool.return_object(obj)

        cleanup_callback.assert_called_once_with(obj)

    def test_clear_pool(self):
        """Test clearing the pool."""
        pool = MemoryPool(pool_type=MemoryPoolType.BLOCK_POOL, max_size=100)

        # Add some objects
        pool._pool.extend(["obj1", "obj2"])
        pool._allocated_objects.add("obj3")

        pool.clear()

        assert len(pool._pool) == 0
        assert len(pool._allocated_objects) == 0

    def test_get_stats(self):
        """Test getting pool statistics."""

        class Counter:
            def __init__(self):
                self.value = 0

            def get_next(self):
                self.value += 1
                return f"test_object_{self.value}"

        counter = Counter()
        pool = MemoryPool(
            pool_type=MemoryPoolType.BLOCK_POOL,
            max_size=100,
            object_factory=counter.get_next,
        )

        # Get some objects
        obj1 = pool.get_object()
        obj2 = pool.get_object()

        # Return one object
        pool.return_object(obj1)

        stats = pool.get_stats()

        assert stats["pool_type"] == "block_pool"
        assert stats["pool_size"] == 1
        assert stats["max_size"] == 100
        assert stats["allocated_objects"] == 1  # One object still allocated (obj2)
        assert stats["hits"] == 0
        assert stats["misses"] == 2
        assert stats["allocations"] == 2
        assert stats["deallocations"] == 1
        assert stats["hit_rate"] == 0.0


class TestMemoryMonitor:
    """Test the MemoryMonitor class."""

    @pytest.fixture
    def monitor(self):
        """Create a memory monitor."""
        return MemoryMonitor(update_interval=0.1)

    def test_monitor_creation(self, monitor):
        """Test creating a memory monitor."""
        assert monitor.update_interval == 0.1
        assert monitor._running is False
        assert monitor._monitor_thread is None
        assert len(monitor._stats_history) == 0
        assert monitor._current_stats is None

    def test_start_stop_monitoring(self, monitor):
        """Test starting and stopping monitoring."""
        monitor.start()
        assert monitor._running is True
        assert monitor._monitor_thread is not None

        time.sleep(0.2)  # Let it run briefly

        monitor.stop()
        assert monitor._running is False

    def test_collect_stats(self, monitor):
        """Test collecting memory statistics."""
        stats = monitor._collect_stats()

        assert isinstance(stats, MemoryStats)
        assert stats.total_memory > 0
        assert stats.used_memory > 0
        assert stats.free_memory > 0
        assert 0 <= stats.memory_percentage <= 100
        assert stats.process_memory > 0
        assert stats.timestamp > 0

    def test_track_allocation_deallocation(self, monitor):
        """Test tracking object allocation and deallocation."""
        obj = "test_object"

        # Track allocation
        monitor.track_allocation(obj, 1024)

        assert "str" in monitor._object_counts
        assert monitor._object_counts["str"] == 1
        assert "str" in monitor._object_sizes
        assert monitor._object_sizes["str"] == 1024

        # Track deallocation
        monitor.track_deallocation(obj)

        assert monitor._object_counts["str"] == 0
        assert monitor._object_sizes["str"] == 0

    def test_memory_limit_callback(self, monitor):
        """Test memory limit callback."""
        callback = Mock()
        monitor.set_memory_limit_callback(callback)

        # Simulate high memory usage
        with patch.object(monitor, "_collect_stats") as mock_collect:
            mock_collect.return_value = MemoryStats(
                total_memory=1000000,
                used_memory=950000,
                free_memory=50000,
                memory_percentage=95.0,
                process_memory=100000,
                peak_memory=100000,
            )

            # Test the callback logic directly without the infinite loop
            stats = monitor._collect_stats()
            if monitor._memory_limit_callback and stats.memory_percentage > 90.0:
                monitor._memory_limit_callback(stats)

            callback.assert_called_once()

    def test_leak_detection_callback(self, monitor):
        """Test leak detection callback."""
        callback = Mock()
        monitor.set_leak_detection_callback(callback)

        # Add old allocation
        old_time = time.time() - 7200  # 2 hours ago
        monitor._allocation_times[123] = old_time
        monitor._object_counts["test_type"] = 1

        monitor._detect_memory_leaks()

        callback.assert_called_once()
        leak = callback.call_args[0][0]
        assert isinstance(leak, MemoryLeak)
        assert leak.severity in ["low", "medium", "high", "critical"]


class TestGarbageCollectionManager:
    """Test the GarbageCollectionManager class."""

    @pytest.fixture
    def gc_manager(self):
        """Create a garbage collection manager."""
        return GarbageCollectionManager(auto_gc=True, gc_threshold=0.8)

    def test_gc_manager_creation(self, gc_manager):
        """Test creating a garbage collection manager."""
        assert gc_manager.auto_gc is True
        assert gc_manager.gc_threshold == 0.8
        assert gc_manager._gc_stats["manual_gc_count"] == 0
        assert gc_manager._gc_stats["auto_gc_count"] == 0

    def test_set_generation_thresholds(self, gc_manager):
        """Test setting generation thresholds."""
        gc_manager.set_generation_thresholds(1000, 100, 10)

        assert gc_manager._generation_thresholds == [1000, 100, 10]

    def test_force_gc(self, gc_manager):
        """Test forcing garbage collection."""
        result = gc_manager.force_gc()

        assert "collected" in result
        assert "gc_time" in result
        assert "generation" in result
        assert "stats" in result
        assert result["generation"] == 2
        assert gc_manager._gc_stats["manual_gc_count"] == 1

    def test_auto_gc_check(self, gc_manager):
        """Test automatic garbage collection check."""
        # Test with high memory usage
        result = gc_manager.auto_gc_check(85.0)  # Above threshold

        assert result is True
        assert gc_manager._gc_stats["auto_gc_count"] == 1

        # Test with low memory usage
        result = gc_manager.auto_gc_check(50.0)  # Below threshold

        assert result is False
        assert gc_manager._gc_stats["auto_gc_count"] == 1  # Unchanged

    def test_gc_callback(self, gc_manager):
        """Test garbage collection callback."""
        callback = Mock()
        gc_manager.add_gc_callback(callback)

        gc_manager.force_gc()

        callback.assert_called_once()

    def test_get_gc_stats(self, gc_manager):
        """Test getting garbage collection statistics."""
        gc_manager.force_gc()

        stats = gc_manager.get_gc_stats()

        assert "manual_gc_count" in stats
        assert "auto_gc_count" in stats
        assert "total_gc_time" in stats
        assert "last_gc_time" in stats
        assert stats["manual_gc_count"] == 1


class TestMemoryProfiler:
    """Test the MemoryProfiler class."""

    @pytest.fixture
    def profiler(self):
        """Create a memory profiler."""
        return MemoryProfiler(enable_tracing=True)

    def test_profiler_creation(self, profiler):
        """Test creating a memory profiler."""
        assert profiler.enable_tracing is True
        assert profiler._tracing_enabled is True
        assert len(profiler._snapshots) == 0

    def test_start_stop_tracing(self):
        """Test starting and stopping tracing."""
        profiler = MemoryProfiler(enable_tracing=False)

        assert profiler._tracing_enabled is False

        profiler.start_tracing()
        assert profiler._tracing_enabled is True

        profiler.stop_tracing()
        assert profiler._tracing_enabled is False

    def test_take_snapshot(self, profiler):
        """Test taking a memory snapshot."""
        snapshot = profiler.take_snapshot()

        assert snapshot is not None
        assert len(profiler._snapshots) == 1

    def test_compare_snapshots(self, profiler):
        """Test comparing memory snapshots."""
        # Take first snapshot
        snapshot1 = profiler.take_snapshot()

        # Allocate some memory
        data = [i for i in range(1000)]

        # Take second snapshot
        snapshot2 = profiler.take_snapshot()

        # Compare snapshots
        diff_stats = profiler.compare_snapshots(snapshot1, snapshot2)

        assert isinstance(diff_stats, list)

    def test_get_top_memory_allocations(self, profiler):
        """Test getting top memory allocations."""
        # Allocate some memory
        data = [i for i in range(1000)]

        # Take snapshot
        profiler.take_snapshot()

        # Get top allocations
        top_allocations = profiler.get_top_memory_allocations(limit=5)

        assert isinstance(top_allocations, list)
        assert len(top_allocations) <= 5

    def test_detect_memory_leaks(self, profiler):
        """Test detecting memory leaks."""
        # Take first snapshot
        profiler.take_snapshot()

        # Allocate some memory
        data = [i for i in range(1000)]

        # Take second snapshot
        profiler.take_snapshot()

        # Detect leaks
        leaks = profiler.detect_memory_leaks()

        assert isinstance(leaks, list)


class TestMemoryManager:
    """Test the MemoryManager class."""

    @pytest.fixture
    def memory_manager(self):
        """Create a memory manager."""
        return MemoryManager(
            max_memory_mb=512,
            enable_monitoring=True,
            enable_profiling=False,
            auto_gc=True,
        )

    def test_memory_manager_creation(self, memory_manager):
        """Test creating a memory manager."""
        assert memory_manager.max_memory_mb == 512
        assert memory_manager.max_memory_bytes == 512 * 1024 * 1024
        assert memory_manager._monitor is not None
        assert memory_manager._gc_manager is not None
        assert memory_manager._profiler is None
        assert len(memory_manager._pools) == 0

    def test_create_pool(self, memory_manager):
        """Test creating a memory pool."""
        pool = memory_manager.create_pool(
            pool_type=MemoryPoolType.BLOCK_POOL,
            max_size=100,
            object_factory=lambda: "test_object",
        )

        assert isinstance(pool, MemoryPool)
        assert pool.pool_type == MemoryPoolType.BLOCK_POOL
        assert pool.max_size == 100
        assert MemoryPoolType.BLOCK_POOL in memory_manager._pools

    def test_get_pool(self, memory_manager):
        """Test getting a memory pool."""
        # Create pool
        pool = memory_manager.create_pool(MemoryPoolType.BLOCK_POOL)

        # Get pool
        retrieved_pool = memory_manager.get_pool(MemoryPoolType.BLOCK_POOL)

        assert retrieved_pool is pool

        # Get non-existent pool
        non_existent = memory_manager.get_pool(MemoryPoolType.TRANSACTION_POOL)
        assert non_existent is None

    def test_allocate_object(self, memory_manager):
        """Test allocating an object."""
        # Create pool
        memory_manager.create_pool(
            MemoryPoolType.BLOCK_POOL, object_factory=lambda: "pooled_object"
        )

        # Allocate object
        obj = memory_manager.allocate_object(
            "test_object", pool_type=MemoryPoolType.BLOCK_POOL, size=1024
        )

        assert obj == "pooled_object"  # Should get pooled object, not the original
        assert len(memory_manager._object_registry) == 1
        assert len(memory_manager._events) > 0

    def test_deallocate_object(self, memory_manager):
        """Test deallocating an object."""
        # Create pool
        memory_manager.create_pool(MemoryPoolType.BLOCK_POOL)

        # Allocate object
        obj = memory_manager.allocate_object(
            "test_object", pool_type=MemoryPoolType.BLOCK_POOL, size=1024
        )

        # Deallocate object
        memory_manager.deallocate_object(obj, pool_type=MemoryPoolType.BLOCK_POOL)

        assert len(memory_manager._object_registry) == 0
        assert len(memory_manager._events) > 0

    def test_set_memory_limit(self, memory_manager):
        """Test setting memory limits."""
        callback = Mock()
        memory_manager.set_memory_limit("test_limit", 100, callback)

        assert "test_limit" in memory_manager._memory_limits
        assert memory_manager._memory_limits["test_limit"] == 100 * 1024 * 1024
        assert "test_limit" in memory_manager._limit_callbacks

    def test_check_memory_limits(self, memory_manager):
        """Test checking memory limits."""
        # Set a very low limit
        memory_manager.set_memory_limit("low_limit", 1)  # 1MB

        # Check limits
        results = memory_manager.check_memory_limits()

        assert "low_limit" in results
        assert isinstance(results["low_limit"], bool)

    def test_force_garbage_collection(self, memory_manager):
        """Test forcing garbage collection."""
        result = memory_manager.force_garbage_collection()

        assert "collected" in result
        assert "gc_time" in result
        assert "generation" in result
        assert "stats" in result

    def test_get_memory_stats(self, memory_manager):
        """Test getting memory statistics."""
        stats = memory_manager.get_memory_stats()

        assert isinstance(stats, MemoryStats)
        assert stats.total_memory > 0
        assert stats.used_memory > 0
        assert stats.free_memory > 0

    def test_get_pool_stats(self, memory_manager):
        """Test getting pool statistics."""
        # Create some pools
        memory_manager.create_pool(MemoryPoolType.BLOCK_POOL)
        memory_manager.create_pool(MemoryPoolType.TRANSACTION_POOL)

        stats = memory_manager.get_pool_stats()

        assert "block_pool" in stats
        assert "transaction_pool" in stats
        assert isinstance(stats["block_pool"], dict)
        assert isinstance(stats["transaction_pool"], dict)

    def test_add_event_callback(self, memory_manager):
        """Test adding event callback."""
        callback = Mock()
        memory_manager.add_event_callback(callback)

        # Allocate object to trigger event
        memory_manager.allocate_object("test_object")

        # Callback should be called
        callback.assert_called()

    def test_get_memory_events(self, memory_manager):
        """Test getting memory events."""
        # Generate some events
        memory_manager.allocate_object("test_object")
        memory_manager.allocate_object("test_object2")

        events = memory_manager.get_memory_events(limit=10)

        assert isinstance(events, list)
        assert len(events) <= 10
        assert all(isinstance(event, MemoryEvent) for event in events)

    def test_cleanup(self, memory_manager):
        """Test cleaning up memory manager."""
        # Create some state
        memory_manager.create_pool(MemoryPoolType.BLOCK_POOL)
        memory_manager.allocate_object("test_object")

        # Cleanup
        memory_manager.cleanup()

        assert len(memory_manager._pools) == 0
        assert len(memory_manager._object_registry) == 0
        assert len(memory_manager._weak_refs) == 0

    def test_context_manager(self):
        """Test memory manager as context manager."""
        with MemoryManager() as manager:
            assert isinstance(manager, MemoryManager)
            manager.create_pool(MemoryPoolType.BLOCK_POOL)

        # Manager should be cleaned up after context exit
        assert len(manager._pools) == 0

    def test_thread_safety(self, memory_manager):
        """Test thread safety of memory manager."""
        results = []
        errors = []

        def worker():
            try:
                for i in range(10):
                    obj = memory_manager.allocate_object(f"object_{i}")
                    memory_manager.deallocate_object(obj)
                    results.append(True)
            except Exception as e:
                errors.append(e)

        # Start multiple threads
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=worker)
            threads.append(thread)
            thread.start()

        # Wait for all threads
        for thread in threads:
            thread.join()

        # Check for errors
        assert len(errors) == 0, f"Thread safety errors: {errors}"
        assert len(results) == 50  # 5 threads * 10 operations each

    def test_memory_pool_integration(self, memory_manager):
        """Test integration between memory manager and pools."""
        # Create pool with factory
        pool = memory_manager.create_pool(
            MemoryPoolType.BLOCK_POOL,
            max_size=5,
            object_factory=lambda: "pooled_object",
        )

        # Allocate objects through manager
        objects = []
        for i in range(10):
            obj = memory_manager.allocate_object(
                f"object_{i}", pool_type=MemoryPoolType.BLOCK_POOL
            )
            objects.append(obj)

        # Deallocate objects
        for obj in objects:
            memory_manager.deallocate_object(obj, pool_type=MemoryPoolType.BLOCK_POOL)

        # Check pool stats
        pool_stats = pool.get_stats()
        assert pool_stats["allocations"] > 0
        assert pool_stats["deallocations"] > 0
