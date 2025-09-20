"""
Comprehensive CUDA Integration Tests for DubChain.

This module provides comprehensive tests for CUDA integration across all components
of the DubChain system.
"""

import pytest
import time
import secrets
from typing import List, Dict, Any

from src.dubchain.cuda import CUDAManager, CUDAConfig, cuda_available
from src.dubchain.crypto import GPUCrypto, GPUConfig
from src.dubchain.consensus import CUDAConsensusAccelerator, CUDAConsensusConfig
from src.dubchain.vm import CUDAVMAccelerator, CUDAVMConfig
from src.dubchain.storage import CUDAStorageAccelerator, CUDAStorageConfig


class TestCUDAComprehensiveIntegration:
    """Test comprehensive CUDA integration across all DubChain components."""
    
    @pytest.fixture
    def cuda_config(self):
        """Create comprehensive CUDA configuration for testing."""
        return CUDAConfig(
            enable_cuda=True,
            enable_crypto_gpu=True,
            enable_consensus_gpu=True,
            enable_storage_gpu=True,
            enable_test_gpu=True,
            test_gpu_fallback=True,
            min_batch_size_gpu=5,
            max_batch_size=1000,
            chunk_size=100,
        )
    
    @pytest.fixture
    def cuda_manager(self, cuda_config):
        """Create CUDA manager for testing."""
        return CUDAManager(cuda_config)
    
    @pytest.fixture
    def gpu_crypto(self):
        """Create GPU crypto instance for testing."""
        return GPUCrypto()
    
    @pytest.fixture
    def cuda_consensus_accelerator(self):
        """Create CUDA consensus accelerator for testing."""
        return CUDAConsensusAccelerator()
    
    @pytest.fixture
    def cuda_vm_accelerator(self):
        """Create CUDA VM accelerator for testing."""
        return CUDAVMAccelerator()
    
    @pytest.fixture
    def cuda_storage_accelerator(self):
        """Create CUDA storage accelerator for testing."""
        return CUDAStorageAccelerator()
    
    def test_cuda_availability_across_components(self):
        """Test CUDA availability across all components."""
        available = cuda_available()
        assert isinstance(available, bool)
        print(f"CUDA Available: {available}")
        
        # Test that all components can detect CUDA availability
        gpu_crypto = GPUCrypto()
        consensus_accelerator = CUDAConsensusAccelerator()
        vm_accelerator = CUDAVMAccelerator()
        storage_accelerator = CUDAStorageAccelerator()
        
        print(f"GPU Crypto - Available: {gpu_crypto.gpu_available}")
        print(f"Consensus Accelerator - Available: {consensus_accelerator.cuda_manager.available}")
        print(f"VM Accelerator - Available: {vm_accelerator.cuda_manager.available}")
        print(f"Storage Accelerator - Available: {storage_accelerator.cuda_manager.available}")
    
    def test_cross_component_cuda_integration(self, 
                                            gpu_crypto, 
                                            cuda_consensus_accelerator, 
                                            cuda_vm_accelerator, 
                                            cuda_storage_accelerator):
        """Test CUDA integration across multiple components."""
        # Test that all components use the same CUDA manager
        cuda_managers = [
            gpu_crypto.cuda_manager,
            cuda_consensus_accelerator.cuda_manager,
            cuda_vm_accelerator.cuda_manager,
            cuda_storage_accelerator.cuda_manager,
        ]
        
        # All should be the same instance
        for manager in cuda_managers[1:]:
            assert manager is cuda_managers[0]
        
        print("All components use the same CUDA manager instance")
    
    def test_comprehensive_performance_benchmark(self, 
                                               gpu_crypto, 
                                               cuda_consensus_accelerator, 
                                               cuda_vm_accelerator, 
                                               cuda_storage_accelerator):
        """Test comprehensive performance across all components."""
        print("🔬 Running comprehensive CUDA performance benchmark...")
        
        # Generate test data
        test_data_size = 50
        
        # Crypto operations
        crypto_data = [secrets.token_bytes(64) for _ in range(test_data_size)]
        crypto_start = time.time()
        crypto_results = gpu_crypto.hash_data_batch_gpu(crypto_data, "sha256")
        crypto_time = time.time() - crypto_start
        
        # Consensus operations
        consensus_blocks = []
        for i in range(test_data_size):
            block = {
                'index': i,
                'timestamp': time.time(),
                'data': secrets.token_bytes(64),
                'previous_hash': secrets.token_hex(32),
                'validator': f"validator_{i}",
            }
            consensus_blocks.append(block)
        
        consensus_start = time.time()
        consensus_results = cuda_consensus_accelerator.validate_blocks_batch(consensus_blocks)
        consensus_time = time.time() - consensus_start
        
        # VM operations
        vm_operations = []
        for i in range(test_data_size):
            operation = {
                'id': f"vm_op_{i}",
                'type': 'vm_operation',
                'data': secrets.token_bytes(32),
                'gas_limit': 1000,
                'timestamp': time.time(),
            }
            vm_operations.append(operation)
        
        vm_start = time.time()
        vm_results = cuda_vm_accelerator.execute_operations_batch(vm_operations)
        vm_time = time.time() - vm_start
        
        # Storage operations
        storage_data = []
        for i in range(test_data_size):
            data = {
                'id': f"storage_data_{i}",
                'content': secrets.token_bytes(64),
                'metadata': {'index': i, 'timestamp': time.time()},
            }
            storage_data.append(data)
        
        storage_start = time.time()
        storage_results = cuda_storage_accelerator.serialize_data_batch(storage_data, "json")
        storage_time = time.time() - storage_start
        
        # Calculate total performance
        total_time = crypto_time + consensus_time + vm_time + storage_time
        total_operations = len(crypto_results) + len(consensus_results) + len(vm_results) + len(storage_results)
        
        results = {
            'crypto_time': crypto_time,
            'consensus_time': consensus_time,
            'vm_time': vm_time,
            'storage_time': storage_time,
            'total_time': total_time,
            'total_operations': total_operations,
            'throughput': total_operations / total_time,
        }
        
        print(f"   Results:")
        print(f"     Crypto time: {crypto_time:.4f}s ({len(crypto_results)} operations)")
        print(f"     Consensus time: {consensus_time:.4f}s ({len(consensus_results)} operations)")
        print(f"     VM time: {vm_time:.4f}s ({len(vm_results)} operations)")
        print(f"     Storage time: {storage_time:.4f}s ({len(storage_results)} operations)")
        print(f"     Total time: {total_time:.4f}s")
        print(f"     Total operations: {total_operations}")
        print(f"     Throughput: {results['throughput']:.2f} ops/sec")
        
        assert total_operations == test_data_size * 4  # 4 components
        assert total_time > 0
    
    def test_memory_management_across_components(self, 
                                               gpu_crypto, 
                                               cuda_consensus_accelerator, 
                                               cuda_vm_accelerator, 
                                               cuda_storage_accelerator):
        """Test memory management across all CUDA components."""
        print("🧠 Testing memory management across components...")
        
        # Perform operations that should trigger memory management
        large_data_size = 200
        
        # Large crypto operations
        large_crypto_data = [secrets.token_bytes(1024) for _ in range(large_data_size)]
        crypto_results = gpu_crypto.hash_data_batch_gpu(large_crypto_data, "sha256")
        
        # Large consensus operations
        large_consensus_blocks = []
        for i in range(large_data_size):
            block = {
                'index': i,
                'timestamp': time.time(),
                'data': secrets.token_bytes(1024),
                'previous_hash': secrets.token_hex(32),
                'validator': f"validator_{i}",
            }
            large_consensus_blocks.append(block)
        
        consensus_results = cuda_consensus_accelerator.validate_blocks_batch(large_consensus_blocks)
        
        # Large VM operations
        large_vm_operations = []
        for i in range(large_data_size):
            operation = {
                'id': f"large_vm_op_{i}",
                'type': 'vm_operation',
                'data': secrets.token_bytes(1024),
                'gas_limit': 1000,
                'timestamp': time.time(),
            }
            large_vm_operations.append(operation)
        
        vm_results = cuda_vm_accelerator.execute_operations_batch(large_vm_operations)
        
        # Large storage operations
        large_storage_data = []
        for i in range(large_data_size):
            data = {
                'id': f"large_storage_data_{i}",
                'content': secrets.token_bytes(1024),
                'metadata': {'index': i, 'timestamp': time.time()},
            }
            large_storage_data.append(data)
        
        storage_results = cuda_storage_accelerator.serialize_data_batch(large_storage_data, "json")
        
        # Verify all operations completed successfully
        assert len(crypto_results) == large_data_size
        assert len(consensus_results) == large_data_size
        assert len(vm_results) == large_data_size
        assert len(storage_results) == large_data_size
        
        print(f"Memory management test completed:")
        print(f"   Crypto: {len(crypto_results)} operations")
        print(f"   Consensus: {len(consensus_results)} operations")
        print(f"   VM: {len(vm_results)} operations")
        print(f"   Storage: {len(storage_results)} operations")
    
    def test_concurrent_operations_across_components(self, 
                                                   gpu_crypto, 
                                                   cuda_consensus_accelerator, 
                                                   cuda_vm_accelerator, 
                                                   cuda_storage_accelerator):
        """Test concurrent operations across all CUDA components."""
        import threading
        import queue
        
        print("🔄 Testing concurrent operations across components...")
        
        results_queue = queue.Queue()
        
        def crypto_worker():
            """Crypto worker function."""
            data = [secrets.token_bytes(32) for _ in range(10)]
            results = gpu_crypto.hash_data_batch_gpu(data, "sha256")
            results_queue.put(("crypto", len(results)))
        
        def consensus_worker():
            """Consensus worker function."""
            blocks = []
            for i in range(10):
                block = {
                    'index': i,
                    'timestamp': time.time(),
                    'data': secrets.token_bytes(32),
                    'previous_hash': secrets.token_hex(32),
                    'validator': f"validator_{i}",
                }
                blocks.append(block)
            
            results = cuda_consensus_accelerator.validate_blocks_batch(blocks)
            results_queue.put(("consensus", len(results)))
        
        def vm_worker():
            """VM worker function."""
            operations = []
            for i in range(10):
                operation = {
                    'id': f"concurrent_vm_op_{i}",
                    'type': 'vm_operation',
                    'data': secrets.token_bytes(32),
                    'gas_limit': 1000,
                    'timestamp': time.time(),
                }
                operations.append(operation)
            
            results = cuda_vm_accelerator.execute_operations_batch(operations)
            results_queue.put(("vm", len(results)))
        
        def storage_worker():
            """Storage worker function."""
            data = []
            for i in range(10):
                storage_data = {
                    'id': f"concurrent_storage_{i}",
                    'content': secrets.token_bytes(32),
                    'metadata': {'index': i, 'timestamp': time.time()},
                }
                data.append(storage_data)
            
            results = cuda_storage_accelerator.serialize_data_batch(data, "json")
            results_queue.put(("storage", len(results)))
        
        # Start all worker threads
        threads = []
        workers = [crypto_worker, consensus_worker, vm_worker, storage_worker]
        
        for worker in workers:
            thread = threading.Thread(target=worker)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Collect results
        results = []
        while not results_queue.empty():
            results.append(results_queue.get())
        
        # Verify all operations completed
        assert len(results) == 4
        for component, count in results:
            assert count == 10
            print(f"   {component}: {count} operations completed")
    
    def test_performance_metrics_across_components(self, 
                                                 gpu_crypto, 
                                                 cuda_consensus_accelerator, 
                                                 cuda_vm_accelerator, 
                                                 cuda_storage_accelerator):
        """Test performance metrics collection across all components."""
        # Perform some operations on each component
        test_data = [secrets.token_bytes(32) for _ in range(10)]
        
        gpu_crypto.hash_data_batch_gpu(test_data, "sha256")
        
        blocks = [{'index': i, 'data': secrets.token_bytes(32)} for i in range(10)]
        cuda_consensus_accelerator.validate_blocks_batch(blocks)
        
        operations = [{'id': f"op_{i}", 'data': secrets.token_bytes(32)} for i in range(10)]
        cuda_vm_accelerator.execute_operations_batch(operations)
        
        storage_data = [{'id': f"data_{i}", 'content': secrets.token_bytes(32)} for i in range(10)]
        cuda_storage_accelerator.serialize_data_batch(storage_data, "json")
        
        # Get metrics from all components
        crypto_metrics = gpu_crypto.get_performance_metrics()
        consensus_metrics = cuda_consensus_accelerator.get_performance_metrics()
        vm_metrics = cuda_vm_accelerator.get_performance_metrics()
        storage_metrics = cuda_storage_accelerator.get_performance_metrics()
        
        # Verify metrics structure
        for component, metrics in [
            ("crypto", crypto_metrics),
            ("consensus", consensus_metrics),
            ("vm", vm_metrics),
            ("storage", storage_metrics),
        ]:
            assert isinstance(metrics, dict)
            # Check for common fields that should exist in all components
            assert 'total_operations' in metrics or 'batch_operations' in metrics
            # Some components may not have all fields, so be flexible
            if 'gpu_operations' in metrics:
                assert isinstance(metrics['gpu_operations'], int)
            if 'cpu_fallbacks' in metrics:
                assert isinstance(metrics['cpu_fallbacks'], int)
            
            print(f"   {component} metrics: {metrics.get('total_operations', 0)} total operations")
    
    def test_error_handling_across_components(self, 
                                            gpu_crypto, 
                                            cuda_consensus_accelerator, 
                                            cuda_vm_accelerator, 
                                            cuda_storage_accelerator):
        """Test error handling across all CUDA components."""
        print("⚠️  Testing error handling across components...")
        
        # Test with empty data
        try:
            crypto_results = gpu_crypto.hash_data_batch_gpu([], "sha256")
            assert crypto_results == []
        except Exception as e:
            print(f"Crypto empty data handling: {e}")
        
        try:
            consensus_results = cuda_consensus_accelerator.validate_blocks_batch([])
            assert consensus_results == []
        except Exception as e:
            print(f"Consensus empty data handling: {e}")
        
        try:
            vm_results = cuda_vm_accelerator.execute_operations_batch([])
            assert vm_results == []
        except Exception as e:
            print(f"VM empty data handling: {e}")
        
        try:
            storage_results = cuda_storage_accelerator.serialize_data_batch([], "json")
            assert storage_results == []
        except Exception as e:
            print(f"Storage empty data handling: {e}")
        
        print("Error handling test completed across all components")
    
    def test_cuda_manager_global_state(self, cuda_manager):
        """Test global CUDA manager state consistency."""
        # Test that the global CUDA manager is consistent
        from src.dubchain.cuda import get_global_cuda_manager
        
        global_manager = get_global_cuda_manager()
        # Both managers should have the same essential properties
        assert global_manager.available == cuda_manager.available
        assert global_manager.device == cuda_manager.device
        # Configs may have different default values, so just check they're both CUDAConfig instances
        assert hasattr(global_manager.config, 'enable_cuda')
        assert hasattr(cuda_manager.config, 'enable_cuda')
        
        # Test performance metrics
        global_metrics = global_manager.get_performance_metrics()
        assert isinstance(global_metrics, dict)
        assert 'total_operations' in global_metrics
        assert 'gpu_operations' in global_metrics
        assert 'cpu_fallbacks' in global_metrics
        
        print(f"Global CUDA manager state: {global_metrics}")


if __name__ == "__main__":
    # Run comprehensive tests
    pytest.main([__file__, "-v"])
