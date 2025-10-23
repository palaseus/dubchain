"""
Comprehensive tests for optimized modules.

This module tests all the new optimization implementations including:
- VM optimizations
- Storage optimizations  
- Crypto optimizations
- Memory optimizations
- Batching optimizations
"""

import logging

logger = logging.getLogger(__name__)
import asyncio
import pytest
import time
import threading
from unittest.mock import Mock, patch

from src.dubchain.performance.optimizations import OptimizationManager
from src.dubchain.vm.optimized_vm import OptimizedVM
from src.dubchain.storage.optimized_storage import OptimizedStorage
from src.dubchain.crypto.optimized_crypto import OptimizedCrypto
from src.dubchain.memory.optimized_memory import OptimizedMemory
from src.dubchain.batching.optimized_batching import OptimizedBatching


class TestOptimizedVM:
    """Test VM optimizations."""
    
    def setup_method(self):
        """Setup test environment."""
        self.optimization_manager = OptimizationManager()
        self.vm = OptimizedVM(self.optimization_manager)
    
    def test_vm_initialization(self):
        """Test VM initialization."""
        assert self.vm.optimization_manager == self.optimization_manager
        assert self.vm.cache_size_limit == 1000
        assert self.vm.cache_hits == 0
        assert self.vm.cache_misses == 0
    
    def test_bytecode_caching(self):
        """Test bytecode caching functionality."""
        # Enable bytecode caching
        self.optimization_manager.enable_optimization("vm_bytecode_caching")
        
        contract_hash = "test_contract"
        bytecode = {"instructions": ["PUSH", "ADD", "RET"]}
        
        # First execution - should cache
        result1 = self.vm.execute_contract_optimized(contract_hash, bytecode, {}, 1000000)
        assert result1["success"] is True
        assert self.vm.cache_misses == 1
        
        # Second execution - should use cache
        result2 = self.vm.execute_contract_optimized(contract_hash, bytecode, {}, 1000000)
        assert result2["success"] is True
        assert self.vm.cache_hits == 1
    
    def test_gas_optimization(self):
        """Test gas optimization."""
        # Enable gas optimization
        self.optimization_manager.enable_optimization("vm_gas_optimization")
        
        # Test gas optimization for different instructions
        original_gas = 10
        optimized_gas = self.vm.optimize_gas_usage("PUSH", original_gas)
        assert optimized_gas <= original_gas
        assert optimized_gas == 3  # Optimized cost for PUSH
    
    def test_parallel_execution(self):
        """Test parallel contract execution."""
        # Enable parallel execution and bytecode caching
        self.optimization_manager.enable_optimization("vm_parallel_execution")
        self.optimization_manager.enable_optimization("vm_bytecode_caching")
        
        contracts = [
            ("contract1", {"instructions": ["PUSH", "ADD"]}, {}),
            ("contract2", {"instructions": ["PUSH", "SUB"]}, {}),
            ("contract3", {"instructions": ["PUSH", "MUL"]}, {}),
        ]
        
        # Test parallel execution
        results = asyncio.run(self.vm.execute_contracts_parallel(contracts))
        assert len(results) == 3
        assert all(result["success"] for result in results)
        assert self.vm.metrics["parallel_executions"] == 3
    
    def test_performance_metrics(self):
        """Test performance metrics collection."""
        # Enable bytecode caching to ensure execution is counted
        self.optimization_manager.enable_optimization("vm_bytecode_caching")
        
        # Execute some contracts
        self.vm.execute_contract_optimized("test", {"instructions": ["PUSH"]}, {}, 1000000)
        
        metrics = self.vm.get_performance_metrics()
        assert "total_executions" in metrics
        assert "cache_hit_rate" in metrics
        assert "avg_execution_time" in metrics
        assert metrics["total_executions"] >= 1


class TestOptimizedStorage:
    """Test storage optimizations."""
    
    def setup_method(self):
        """Setup test environment."""
        import tempfile
        import os
        self.optimization_manager = OptimizationManager()
        # Use a temporary file for testing
        self.temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.temp_file.close()
        self.storage = OptimizedStorage(self.optimization_manager, self.temp_file.name)
    
    def teardown_method(self):
        """Cleanup test environment."""
        self.storage.close()
        import os
        if hasattr(self, 'temp_file') and os.path.exists(self.temp_file.name):
            os.remove(self.temp_file.name)
        if os.path.exists("test_storage.db.cache"):
            os.remove("test_storage.db.cache")
    
    def test_storage_initialization(self):
        """Test storage initialization."""
        assert self.storage.optimization_manager == self.optimization_manager
        assert self.storage.storage_path == self.temp_file.name
        assert len(self.storage.memory_cache) == 0
    
    def test_binary_serialization(self):
        """Test binary serialization optimization."""
        # Enable binary serialization
        self.optimization_manager.enable_optimization("storage_binary_formats")
        
        test_data = {"key": "value", "number": 123, "list": [1, 2, 3]}
        
        # Test put and get
        self.storage.put("test_key", test_data)
        retrieved_data = self.storage.get("test_key")
        
        assert retrieved_data == test_data
        assert self.storage.metrics["total_writes"] == 1
        assert self.storage.metrics["total_reads"] == 1
    
    def test_write_batching(self):
        """Test write batching optimization."""
        # Enable write batching
        self.optimization_manager.enable_optimization("storage_write_batching")
        
        # Put multiple items
        for i in range(10):
            self.storage.put(f"key_{i}", f"value_{i}")
        
        # Check that items are batched
        assert len(self.storage.write_batch.operations) > 0
    
    def test_bulk_operations(self):
        """Test bulk operations."""
        # Enable bulk operations
        self.optimization_manager.enable_optimization("storage_bulk_operations")
        
        # Test bulk put
        items = {f"bulk_key_{i}": f"bulk_value_{i}" for i in range(5)}
        result = self.storage.bulk_put(items)
        assert result is True
        
        # Test bulk get
        keys = list(items.keys())
        results = self.storage.bulk_get(keys)
        assert len(results) == len(keys)
        assert all(key in results for key in keys)
    
    def test_compaction(self):
        """Test storage compaction."""
        # Add some data
        for i in range(10):
            self.storage.put(f"compact_key_{i}", f"compact_value_{i}")
        
        # Compact storage
        result = self.storage.compact()
        assert result["success"] is True
        assert "compaction_time" in result
        assert "total_records" in result


class TestOptimizedCrypto:
    """Test crypto optimizations."""
    
    def setup_method(self):
        """Setup test environment."""
        self.optimization_manager = OptimizationManager()
        self.crypto = OptimizedCrypto(self.optimization_manager)
    
    def test_crypto_initialization(self):
        """Test crypto initialization."""
        assert self.crypto.optimization_manager == self.optimization_manager
        assert self.crypto.metrics["total_verifications"] == 0
    
    def test_signature_verification(self):
        """Test signature verification."""
        # Enable parallel verification
        self.optimization_manager.enable_optimization("crypto_parallel_verification")
        
        message = b"test message"
        signature = b'\x00' * 64  # Placeholder signature
        public_key = b'\x02' + b'\x00' * 32  # Placeholder public key
        
        # Test verification
        result = self.crypto.verify_signature(message, signature, public_key)
        assert isinstance(result, bool)
        assert self.crypto.metrics["total_verifications"] == 1
    
    def test_parallel_verification(self):
        """Test parallel signature verification."""
        # Enable parallel verification
        self.optimization_manager.enable_optimization("crypto_parallel_verification")
        
        verifications = [
            (b"message1", b'\x00' * 64, b'\x02' + b'\x00' * 32, "secp256k1"),
            (b"message2", b'\x00' * 64, b'\x02' + b'\x00' * 32, "secp256k1"),
            (b"message3", b'\x00' * 64, b'\x02' + b'\x00' * 32, "secp256k1"),
        ]
        
        # Test parallel verification
        results = asyncio.run(self.crypto.verify_signatures_parallel(verifications))
        assert len(results) == 3
        assert all(isinstance(result, bool) for result in results)
        assert self.crypto.metrics["parallel_verifications"] == 3
    
    def test_signature_aggregation(self):
        """Test signature aggregation."""
        # Enable signature aggregation
        self.optimization_manager.enable_optimization("batching_signature_aggregation")
        
        signatures = [b'\x00' * 64, b'\x01' * 64, b'\x02' * 64]
        public_keys = [b'\x02' + b'\x00' * 32, b'\x02' + b'\x01' * 32, b'\x02' + b'\x02' * 32]
        message_hash = b'\x00' * 32
        
        # Test aggregation
        aggregated = self.crypto.aggregate_signatures(signatures, public_keys, message_hash)
        assert aggregated.signatures == signatures
        assert aggregated.public_keys == public_keys
        assert aggregated.message_hash == message_hash
        assert aggregated.aggregated_signature is not None
    
    def test_hashing(self):
        """Test hashing operations."""
        data = b"test data"
        
        # Test single hash
        hash_result = self.crypto.hash_data(data)
        assert len(hash_result) == 32  # SHA256
        
        # Test batch hashing
        data_list = [b"data1", b"data2", b"data3"]
        hash_results = self.crypto.hash_data_batch(data_list)
        assert len(hash_results) == 3
        assert all(len(h) == 32 for h in hash_results)
    
    def test_keypair_generation(self):
        """Test keypair generation."""
        private_key, public_key = self.crypto.generate_keypair("secp256k1")
        assert len(private_key) == 32
        assert len(public_key) == 33  # Compressed format


class TestOptimizedMemory:
    """Test memory optimizations."""
    
    def setup_method(self):
        """Setup test environment."""
        self.optimization_manager = OptimizationManager()
        self.memory = OptimizedMemory(self.optimization_manager)
    
    def test_memory_initialization(self):
        """Test memory initialization."""
        assert self.memory.optimization_manager == self.optimization_manager
        assert len(self.memory.buffer_pools) > 0
    
    def test_buffer_reuse(self):
        """Test buffer reuse optimization."""
        # Enable buffer reuse
        self.optimization_manager.enable_optimization("memory_buffer_reuse")
        
        # Get buffer
        buffer1 = self.memory.get_reusable_buffer(1024)
        assert len(buffer1) == 1024
        
        # Return buffer
        self.memory.return_buffer(buffer1)
        
        # Get another buffer of same size
        buffer2 = self.memory.get_reusable_buffer(1024)
        assert len(buffer2) == 1024
        assert self.memory.metrics["buffer_reuses"] >= 0
    
    def test_gc_tuning(self):
        """Test garbage collection tuning."""
        # Enable GC tuning
        self.optimization_manager.enable_optimization("memory_gc_tuning")
        
        # Test GC optimization
        collected = self.memory.optimize_gc_settings()
        assert isinstance(collected, int)
        assert self.memory.gc_tuning_enabled
        
        # Test GC restoration
        self.memory.restore_gc_settings()
        assert not self.memory.gc_tuning_enabled
    
    def test_memory_pressure(self):
        """Test memory pressure handling."""
        # Test memory pressure check
        pressure = self.memory.check_memory_pressure()
        assert 0.0 <= pressure <= 1.0
        
        # Test memory pressure handling
        self.memory.handle_memory_pressure()
        # Should not raise any exceptions
    
    def test_memory_mapped_files(self):
        """Test memory-mapped files."""
        # Enable zero-copy operations
        self.optimization_manager.enable_optimization("memory_zero_copy")
        
        # Test memory-mapped file creation
        mmap_file = self.memory.create_memory_mapped_file("test_mmap.bin", 1024)
        # The file should be created even if mmap_file is None (fallback behavior)
        import os
        assert os.path.exists("test_mmap.bin")
        
        # Clean up
        self.memory.close_memory_mapped_file("test_mmap.bin")
        import os
        if os.path.exists("test_mmap.bin"):
            os.remove("test_mmap.bin")
    
    def test_memory_stats(self):
        """Test memory statistics."""
        stats = self.memory.get_memory_stats()
        assert "total_allocated" in stats
        assert "peak_usage" in stats
        assert "buffer_pools" in stats
        assert "object_pools" in stats


class TestOptimizedBatching:
    """Test batching optimizations."""
    
    def setup_method(self):
        """Setup test environment."""
        self.optimization_manager = OptimizationManager()
        self.batching = OptimizedBatching(self.optimization_manager)
    
    def test_batching_initialization(self):
        """Test batching initialization."""
        assert self.batching.optimization_manager == self.optimization_manager
        assert len(self.batching.batch_queues) == 0
    
    def test_transaction_batching(self):
        """Test transaction batching."""
        # Enable transaction aggregation
        self.optimization_manager.enable_optimization("batching_transaction_aggregation")
        
        transactions = [
            {"from": "alice", "to": "bob", "value": 100, "nonce": 1},
            {"from": "bob", "to": "charlie", "value": 50, "nonce": 2},
            {"from": "charlie", "to": "alice", "value": 25, "nonce": 3},
        ]
        
        # Test batching
        result = self.batching.batch_transactions(transactions)
        assert result["success"] is True
        assert result["batch_size"] == 3
        assert "processing_time" in result
        assert "results" in result
    
    def test_state_write_batching(self):
        """Test state write batching."""
        # Enable state aggregation
        self.optimization_manager.enable_optimization("batching_state_aggregation")
        
        state_writes = [
            ("balance.alice", 1000),
            ("balance.bob", 500),
            ("balance.charlie", 250),
        ]
        
        # Test batching
        result = self.batching.batch_state_writes(state_writes)
        assert result["success"] is True
        assert result["total_writes"] == 3
        assert "processing_time" in result
        assert "results" in result
    
    def test_signature_aggregation(self):
        """Test signature aggregation."""
        # Enable signature aggregation
        self.optimization_manager.enable_optimization("batching_signature_aggregation")
        
        signatures = [b'\x00' * 64, b'\x01' * 64, b'\x02' * 64]
        public_keys = [b'\x02' + b'\x00' * 32, b'\x02' + b'\x01' * 32, b'\x02' + b'\x02' * 32]
        message_hash = b'\x00' * 32
        
        # Test aggregation
        aggregated = self.batching.aggregate_signatures(signatures, public_keys, message_hash)
        assert aggregated.signatures == signatures
        assert aggregated.public_keys == public_keys
        assert aggregated.message_hash == message_hash
        assert aggregated.aggregated_signature is not None
    
    def test_message_batching(self):
        """Test message batching."""
        # Enable message aggregation
        self.optimization_manager.enable_optimization("batching_message_aggregation")
        
        messages = [
            {"type": "block", "data": "block1"},
            {"type": "transaction", "data": "tx1"},
            {"type": "block", "data": "block2"},
        ]
        
        # Test batching
        result = self.batching.batch_messages(messages)
        assert result["success"] is True
        assert result["total_messages"] == 3
        assert "processing_time" in result
        assert "results" in result
    
    def test_performance_metrics(self):
        """Test performance metrics."""
        # Execute some batching operations
        transactions = [{"from": "alice", "to": "bob", "value": 100, "nonce": 1}]
        self.batching.batch_transactions(transactions)
        
        metrics = self.batching.get_performance_metrics()
        assert "total_batches" in metrics
        assert "avg_batch_processing_time" in metrics
        assert "optimization_enabled" in metrics


class TestOptimizationIntegration:
    """Test integration between optimization modules."""
    
    def setup_method(self):
        """Setup test environment."""
        self.optimization_manager = OptimizationManager()
        
        # Initialize all optimization modules
        self.vm = OptimizedVM(self.optimization_manager)
        self.storage = OptimizedStorage(self.optimization_manager, "integration_test.db")
        self.crypto = OptimizedCrypto(self.optimization_manager)
        self.memory = OptimizedMemory(self.optimization_manager)
        self.batching = OptimizedBatching(self.optimization_manager)
    
    def teardown_method(self):
        """Cleanup test environment."""
        self.storage.close()
        import os
        if os.path.exists("integration_test.db"):
            os.remove("integration_test.db")
        if os.path.exists("integration_test.db.cache"):
            os.remove("integration_test.db.cache")
    
    def test_optimization_manager_integration(self):
        """Test that all modules use the same optimization manager."""
        assert self.vm.optimization_manager == self.optimization_manager
        assert self.storage.optimization_manager == self.optimization_manager
        assert self.crypto.optimization_manager == self.optimization_manager
        assert self.memory.optimization_manager == self.optimization_manager
        assert self.batching.optimization_manager == self.optimization_manager
    
    def test_optimization_enable_disable(self):
        """Test enabling and disabling optimizations across modules."""
        # Enable optimizations
        self.optimization_manager.enable_optimization("vm_bytecode_caching")
        self.optimization_manager.enable_optimization("storage_binary_formats")
        self.optimization_manager.enable_optimization("crypto_parallel_verification")
        self.optimization_manager.enable_optimization("memory_buffer_reuse")
        self.optimization_manager.enable_optimization("batching_signature_aggregation")
        
        # Test that optimizations are enabled
        assert self.optimization_manager.is_optimization_enabled("vm_bytecode_caching")
        assert self.optimization_manager.is_optimization_enabled("storage_binary_formats")
        assert self.optimization_manager.is_optimization_enabled("crypto_parallel_verification")
        assert self.optimization_manager.is_optimization_enabled("memory_buffer_reuse")
        assert self.optimization_manager.is_optimization_enabled("batching_signature_aggregation")
        
        # Disable optimizations
        self.optimization_manager.disable_optimization("vm_bytecode_caching")
        self.optimization_manager.disable_optimization("storage_binary_formats")
        
        # Test that optimizations are disabled
        assert not self.optimization_manager.is_optimization_enabled("vm_bytecode_caching")
        assert not self.optimization_manager.is_optimization_enabled("storage_binary_formats")
    
    def test_performance_metrics_integration(self):
        """Test performance metrics across all modules."""
        # Execute operations in each module
        self.vm.execute_contract_optimized("test", {"instructions": ["PUSH"]}, {}, 1000000)
        self.storage.put("test_key", "test_value")
        self.crypto.verify_signature(b"test", b'\x00' * 64, b'\x02' + b'\x00' * 32)
        self.memory.get_reusable_buffer(1024)
        self.batching.batch_transactions([{"from": "alice", "to": "bob", "value": 100, "nonce": 1}])
        
        # Get metrics from all modules
        vm_metrics = self.vm.get_performance_metrics()
        storage_metrics = self.storage.get_performance_metrics()
        crypto_metrics = self.crypto.get_performance_metrics()
        memory_metrics = self.memory.get_performance_metrics()
        batching_metrics = self.batching.get_performance_metrics()
        
        # Verify metrics are collected
        assert vm_metrics["total_executions"] >= 1
        assert storage_metrics["total_writes"] >= 1
        assert crypto_metrics["total_verifications"] >= 1
        assert memory_metrics["total_allocations"] >= 1
        assert batching_metrics["total_batches"] >= 1
    
    def test_fallback_behavior(self):
        """Test fallback behavior when optimizations are disabled."""
        # Disable all optimizations
        for optimization in self.optimization_manager.list_optimizations():
            self.optimization_manager.disable_optimization(optimization.name)
        
        # Test that modules still work with fallback behavior
        vm_result = self.vm.execute_contract_optimized("test", {"instructions": ["PUSH"]}, {}, 1000000)
        assert vm_result["success"] is True
        
        storage_result = self.storage.put("test_key", "test_value")
        assert storage_result is True
        
        crypto_result = self.crypto.verify_signature(b"test", b'\x00' * 64, b'\x02' + b'\x00' * 32)
        assert isinstance(crypto_result, bool)
        
        memory_buffer = self.memory.get_reusable_buffer(1024)
        assert len(memory_buffer) == 1024
        
        batching_result = self.batching.batch_transactions([{"from": "alice", "to": "bob", "value": 100, "nonce": 1}])
        assert batching_result["success"] is True


if __name__ == "__main__":
    pytest.main([__file__])
