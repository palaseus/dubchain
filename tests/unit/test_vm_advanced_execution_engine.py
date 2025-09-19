"""
Unit tests for advanced execution engine.
"""

import time
from dataclasses import dataclass
from unittest.mock import MagicMock, Mock, patch

import pytest

from dubchain.vm.advanced_execution_engine import (
    AdvancedExecutionEngine,
    ExecutionCache,
    ExecutionMetrics,
    ParallelExecutionContext,
    PerformanceMonitor,
)
from dubchain.vm.contract import SmartContract
from dubchain.vm.execution_engine import ExecutionResult


class TestExecutionMetrics:
    """Test ExecutionMetrics class."""

    def test_execution_metrics_creation(self):
        """Test creating execution metrics."""
        metrics = ExecutionMetrics()
        assert metrics.total_executions == 0
        assert metrics.successful_executions == 0
        assert metrics.failed_executions == 0
        assert metrics.average_execution_time == 0.0
        assert metrics.average_gas_used == 0.0
        assert metrics.cache_hits == 0
        assert metrics.cache_misses == 0
        assert metrics.parallel_executions == 0
        assert metrics.optimization_applied == 0
        assert isinstance(metrics.last_updated, float)

    def test_success_rate_calculation(self):
        """Test success rate calculation."""
        metrics = ExecutionMetrics()
        assert metrics.success_rate == 0.0

        metrics.total_executions = 10
        metrics.successful_executions = 7
        assert metrics.success_rate == 0.7

        metrics.total_executions = 0
        assert metrics.success_rate == 0.0

    def test_cache_hit_rate_calculation(self):
        """Test cache hit rate calculation."""
        metrics = ExecutionMetrics()
        assert metrics.cache_hit_rate == 0.0

        metrics.cache_hits = 8
        metrics.cache_misses = 2
        assert metrics.cache_hit_rate == 0.8

        metrics.cache_hits = 0
        metrics.cache_misses = 0
        assert metrics.cache_hit_rate == 0.0


class TestExecutionCache:
    """Test ExecutionCache class."""

    def test_execution_cache_creation(self):
        """Test creating execution cache."""
        cache = ExecutionCache()
        assert len(cache.cache) == 0
        assert cache.max_size == 1000
        assert cache.hit_count == 0
        assert cache.miss_count == 0

    def test_get_cache_key(self):
        """Test cache key generation."""
        cache = ExecutionCache()
        bytecode = b"test_bytecode"
        input_data = b"test_input"
        gas_limit = 100000

        key1 = cache.get_cache_key(bytecode, input_data, gas_limit)
        key2 = cache.get_cache_key(bytecode, input_data, gas_limit)

        assert key1 == key2
        assert isinstance(key1, str)
        assert len(key1) == 64  # SHA256 hex length

    def test_cache_operations(self):
        """Test cache get/put operations."""
        cache = ExecutionCache()
        bytecode = b"test_bytecode"
        input_data = b"test_input"
        gas_limit = 100000
        result = ExecutionResult(
            success=True,
            return_data=b"result",
            gas_used=50000,
            events=[],
            storage_changes={},
        )

        # Test cache miss
        cached_result = cache.get(bytecode, input_data, gas_limit)
        assert cached_result is None
        assert cache.miss_count == 1
        assert cache.hit_count == 0

        # Test cache put
        cache.put(bytecode, input_data, gas_limit, result)
        assert len(cache.cache) == 1

        # Test cache hit
        cached_result = cache.get(bytecode, input_data, gas_limit)
        assert cached_result == result
        assert cache.hit_count == 1
        assert cache.miss_count == 1

    def test_cache_clear(self):
        """Test cache clearing."""
        cache = ExecutionCache()
        bytecode = b"test_bytecode"
        input_data = b"test_input"
        gas_limit = 100000
        result = ExecutionResult(
            success=True,
            return_data=b"result",
            gas_used=50000,
            events=[],
            storage_changes={},
        )

        cache.put(bytecode, input_data, gas_limit, result)
        cache.get(bytecode, input_data, gas_limit)  # Hit
        cache.get(b"other", b"other", 200000)  # Miss

        assert len(cache.cache) == 1
        assert cache.hit_count == 1
        assert cache.miss_count == 1

        cache.clear()
        assert len(cache.cache) == 0
        assert cache.hit_count == 0
        assert cache.miss_count == 0

    def test_cache_size_limit(self):
        """Test cache size limit enforcement."""
        cache = ExecutionCache(max_size=2)

        # Add items up to limit
        cache.put(
            b"bytecode1",
            b"input1",
            100000,
            ExecutionResult(
                success=True,
                return_data=b"result1",
                gas_used=10000,
                events=[],
                storage_changes={},
            ),
        )
        cache.put(
            b"bytecode2",
            b"input2",
            200000,
            ExecutionResult(
                success=True,
                return_data=b"result2",
                gas_used=20000,
                events=[],
                storage_changes={},
            ),
        )

        assert len(cache.cache) == 2

        # Add one more item - should remove oldest
        cache.put(
            b"bytecode3",
            b"input3",
            300000,
            ExecutionResult(
                success=True,
                return_data=b"result3",
                gas_used=30000,
                events=[],
                storage_changes={},
            ),
        )

        assert len(cache.cache) == 2
        # First item should be removed
        assert cache.get(b"bytecode1", b"input1", 100000) is None
        # Other items should still be there
        assert cache.get(b"bytecode2", b"input2", 200000) is not None
        assert cache.get(b"bytecode3", b"input3", 300000) is not None


class TestParallelExecutionContext:
    """Test ParallelExecutionContext class."""

    def test_parallel_execution_context_creation(self):
        """Test creating parallel execution context."""
        context = ParallelExecutionContext(
            thread_id=123,
            bytecode=b"test_bytecode",
            input_data=b"test_input",
            gas_limit=100000,
        )

        assert context.thread_id == 123
        assert context.bytecode == b"test_bytecode"
        assert context.input_data == b"test_input"
        assert context.gas_limit == 100000
        assert context.result is None
        assert context.error is None
        assert isinstance(context.start_time, float)
        assert context.end_time is None


class TestPerformanceMonitor:
    """Test PerformanceMonitor class."""

    def test_performance_monitor_creation(self):
        """Test creating performance monitor."""
        monitor = PerformanceMonitor()
        assert len(monitor.performance_data) == 0
        assert monitor.monitoring_enabled is True

    def test_record_execution_time(self):
        """Test recording execution time."""
        monitor = PerformanceMonitor()
        monitor.record_execution_time("test_operation", 1.5)

        assert "test_operation" in monitor.performance_data
        assert monitor.performance_data["test_operation"] == [1.5]

    def test_record_gas_usage(self):
        """Test recording gas usage."""
        monitor = PerformanceMonitor()
        monitor.record_gas_usage("test_operation", 50000)

        assert "test_operation_gas" in monitor.performance_data
        assert monitor.performance_data["test_operation_gas"] == [50000]

    def test_get_performance_stats(self):
        """Test getting performance statistics."""
        monitor = PerformanceMonitor()
        monitor.record_execution_time("test_operation", 1.0)
        monitor.record_execution_time("test_operation", 2.0)
        monitor.record_execution_time("test_operation", 3.0)

        stats = monitor.get_performance_stats("test_operation")

        assert stats["count"] == 3
        assert stats["min"] == 1.0
        assert stats["max"] == 3.0
        assert stats["avg"] == 2.0
        assert stats["median"] == 2.0

    def test_get_performance_stats_empty(self):
        """Test getting performance statistics for non-existent operation."""
        monitor = PerformanceMonitor()
        stats = monitor.get_performance_stats("non_existent")
        assert stats == {}

    def test_get_all_performance_stats(self):
        """Test getting all performance statistics."""
        monitor = PerformanceMonitor()
        monitor.record_execution_time("op1", 1.0)
        monitor.record_execution_time("op2", 2.0)

        all_stats = monitor.get_all_performance_stats()

        assert "op1" in all_stats
        assert "op2" in all_stats
        assert all_stats["op1"]["count"] == 1
        assert all_stats["op2"]["count"] == 1

    def test_clear_performance_data(self):
        """Test clearing performance data."""
        monitor = PerformanceMonitor()
        monitor.record_execution_time("test_operation", 1.0)

        assert len(monitor.performance_data) > 0
        monitor.clear_performance_data()
        assert len(monitor.performance_data) == 0

    def test_enable_disable_monitoring(self):
        """Test enabling and disabling monitoring."""
        monitor = PerformanceMonitor()

        monitor.disable_monitoring()
        assert monitor.monitoring_enabled is False

        monitor.record_execution_time("test_operation", 1.0)
        assert "test_operation" not in monitor.performance_data

        monitor.enable_monitoring()
        assert monitor.monitoring_enabled is True

        monitor.record_execution_time("test_operation", 1.0)
        assert "test_operation" in monitor.performance_data


class TestAdvancedExecutionEngine:
    """Test AdvancedExecutionEngine class."""

    @pytest.fixture
    def mock_contract(self):
        """Create a mock smart contract."""
        contract = Mock(spec=SmartContract)
        contract.bytecode = b"test_bytecode"
        contract.address = "0x1234567890123456789012345678901234567890"
        return contract

    @pytest.fixture
    def execution_engine(self):
        """Create an advanced execution engine."""
        return AdvancedExecutionEngine()

    def test_advanced_execution_engine_creation(self, execution_engine):
        """Test creating advanced execution engine."""
        assert isinstance(execution_engine.execution_cache, ExecutionCache)
        assert isinstance(execution_engine.metrics, ExecutionMetrics)
        assert execution_engine.parallel_executor is not None
        assert len(execution_engine.active_parallel_executions) == 0
        assert len(execution_engine.optimization_rules) == 4
        assert isinstance(execution_engine.performance_monitor, PerformanceMonitor)

    def test_setup_optimization_rules(self, execution_engine):
        """Test optimization rules setup."""
        rules = execution_engine.optimization_rules

        assert "gas_optimization" in rules
        assert "memory_optimization" in rules
        assert "storage_optimization" in rules
        assert "parallel_optimization" in rules

        # Test that rules are callable
        for rule_name, rule_func in rules.items():
            assert callable(rule_func)

    def test_optimize_gas_usage(self, execution_engine):
        """Test gas usage optimization."""
        bytecode = b"test\x7fbytecode"  # Contains PUSH32 (0x7f)
        optimized = execution_engine._optimize_gas_usage(bytecode)

        # Should replace PUSH32 with PUSH1
        assert b"\x7f" not in optimized
        assert b"\x60" in optimized

    def test_optimize_memory_usage(self, execution_engine):
        """Test memory usage optimization."""
        bytecode = b"test_bytecode"
        optimized = execution_engine._optimize_memory_usage(bytecode)

        # Should return the same bytecode (simplified implementation)
        assert optimized == bytecode

    def test_optimize_storage_usage(self, execution_engine):
        """Test storage usage optimization."""
        bytecode = b"test_bytecode"
        optimized = execution_engine._optimize_storage_usage(bytecode)

        # Should return the same bytecode (simplified implementation)
        assert optimized == bytecode

    def test_optimize_parallel_execution(self, execution_engine):
        """Test parallel execution optimization."""
        bytecode = b"test_bytecode"
        optimized = execution_engine._optimize_parallel_execution(bytecode)

        # Should return the same bytecode (simplified implementation)
        assert optimized == bytecode

    def test_can_execute_parallel(self, execution_engine):
        """Test parallel execution capability check."""
        # Should return False for normal bytecode
        normal_bytecode = b"normal_bytecode"
        assert execution_engine._can_execute_parallel(normal_bytecode) is False

        # Should return True for bytecode with parallel markers
        parallel_bytecode = b"PARALLEL_START_bytecode"
        assert execution_engine._can_execute_parallel(parallel_bytecode) is True

        parallel_bytecode = b"PARALLEL_FORK_bytecode"
        assert execution_engine._can_execute_parallel(parallel_bytecode) is True

    def test_apply_optimizations(self, execution_engine):
        """Test applying optimizations."""
        bytecode = b"test\x7fbytecode"  # Contains PUSH32
        optimized = execution_engine._apply_optimizations(bytecode)

        # Should apply gas optimization (replace PUSH32 with PUSH1)
        assert optimized != bytecode
        assert b"\x7f" not in optimized

    def test_update_metrics(self, execution_engine):
        """Test metrics update."""
        result = ExecutionResult(
            success=True,
            return_data=b"result",
            gas_used=50000,
            events=[],
            storage_changes={},
        )
        execution_time = 1.5

        initial_total = execution_engine.metrics.total_executions
        initial_successful = execution_engine.metrics.successful_executions

        execution_engine._update_metrics(result, execution_time, 100000)

        assert execution_engine.metrics.total_executions == initial_total + 1
        assert execution_engine.metrics.successful_executions == initial_successful + 1
        assert execution_engine.metrics.failed_executions == 0
        assert execution_engine.metrics.average_execution_time > 0
        assert execution_engine.metrics.average_gas_used > 0

    def test_update_metrics_failed_execution(self, execution_engine):
        """Test metrics update for failed execution."""
        result = ExecutionResult(
            success=False,
            return_data=b"",
            gas_used=0,
            events=[],
            storage_changes={},
            error_message="Error",
        )
        execution_time = 0.5

        initial_total = execution_engine.metrics.total_executions
        initial_failed = execution_engine.metrics.failed_executions

        execution_engine._update_metrics(result, execution_time, 100000)

        assert execution_engine.metrics.total_executions == initial_total + 1
        assert execution_engine.metrics.failed_executions == initial_failed + 1
        assert execution_engine.metrics.successful_executions == 0

    def test_get_execution_metrics(self, execution_engine):
        """Test getting execution metrics."""
        metrics = execution_engine.get_execution_metrics()
        assert isinstance(metrics, ExecutionMetrics)
        assert metrics == execution_engine.metrics

    def test_get_cache_metrics(self, execution_engine):
        """Test getting cache metrics."""
        # Add some data to cache
        execution_engine.execution_cache.put(
            b"test",
            b"input",
            100000,
            ExecutionResult(
                success=True,
                return_data=b"result",
                gas_used=50000,
                events=[],
                storage_changes={},
            ),
        )
        execution_engine.execution_cache.get(b"test", b"input", 100000)  # Hit
        execution_engine.execution_cache.get(b"other", b"other", 200000)  # Miss

        cache_metrics = execution_engine.get_cache_metrics()

        assert cache_metrics["cache_size"] == 1
        assert cache_metrics["max_size"] == 1000
        assert cache_metrics["hit_count"] == 1
        assert cache_metrics["miss_count"] == 1
        assert cache_metrics["hit_rate"] == 0.5

    def test_clear_cache(self, execution_engine):
        """Test clearing cache."""
        # Add data to cache
        execution_engine.execution_cache.put(
            b"test",
            b"input",
            100000,
            ExecutionResult(
                success=True,
                return_data=b"result",
                gas_used=50000,
                events=[],
                storage_changes={},
            ),
        )
        assert len(execution_engine.execution_cache.cache) == 1

        execution_engine.clear_cache()
        assert len(execution_engine.execution_cache.cache) == 0

    @patch("dubchain.vm.execution_engine.ExecutionEngine.execute_contract")
    def test_execute_contract_advanced_cached(
        self, mock_execute, execution_engine, mock_contract
    ):
        """Test advanced contract execution with caching."""
        # Setup cache with result
        cached_result = ExecutionResult(
            success=True,
            return_data=b"cached_result",
            gas_used=30000,
            events=[],
            storage_changes={},
        )
        execution_engine.execution_cache.put(
            mock_contract.bytecode, b"test_input", 100000, cached_result
        )

        # Execute with caching enabled
        result = execution_engine.execute_contract_advanced(
            contract=mock_contract,
            input_data=b"test_input",
            gas_limit=100000,
            enable_caching=True,
        )

        # Should return cached result
        assert result == cached_result
        assert execution_engine.metrics.cache_hits == 1
        # Should not call the parent execute_contract method
        mock_execute.assert_not_called()

    @patch("dubchain.vm.execution_engine.ExecutionEngine.execute_contract")
    def test_execute_contract_advanced_no_cache(
        self, mock_execute, execution_engine, mock_contract
    ):
        """Test advanced contract execution without caching."""
        # Setup mock return value
        mock_result = ExecutionResult(
            success=True,
            return_data=b"result",
            gas_used=50000,
            events=[],
            storage_changes={},
        )
        mock_execute.return_value = mock_result

        # Execute with caching disabled
        result = execution_engine.execute_contract_advanced(
            contract=mock_contract,
            input_data=b"test_input",
            gas_limit=100000,
            enable_caching=False,
        )

        # Should call parent method and return result
        assert result.success == mock_result.success
        assert result.return_data == mock_result.return_data
        assert result.gas_used == mock_result.gas_used
        mock_execute.assert_called_once()
        assert execution_engine.metrics.cache_hits == 0

    @patch("dubchain.vm.execution_engine.ExecutionEngine.execute_contract")
    def test_execute_contract_advanced_with_optimizations(
        self, mock_execute, execution_engine, mock_contract
    ):
        """Test advanced contract execution with optimizations."""
        # Setup mock return value
        mock_result = ExecutionResult(
            success=True,
            return_data=b"result",
            gas_used=50000,
            events=[],
            storage_changes={},
        )
        mock_execute.return_value = mock_result

        # Set bytecode with PUSH32 to test optimization
        mock_contract.bytecode = b"test\x7fbytecode"

        # Execute with optimizations enabled
        result = execution_engine.execute_contract_advanced(
            contract=mock_contract,
            input_data=b"test_input",
            gas_limit=100000,
            enable_optimizations=True,
        )

        # Should apply optimizations and cache result
        assert result.success == mock_result.success
        assert result.return_data == mock_result.return_data
        assert result.gas_used == mock_result.gas_used
        assert execution_engine.metrics.optimization_applied == 1
        mock_execute.assert_called_once()

    def test_shutdown(self, execution_engine):
        """Test execution engine shutdown."""
        # Should not raise any exceptions
        execution_engine.shutdown()

        # Verify executor is shutdown
        assert execution_engine.parallel_executor._shutdown is True
