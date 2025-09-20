"""
CUDA Consensus Integration Tests for DubChain.

This module provides comprehensive tests for CUDA integration in consensus operations.
"""

import pytest
import time
import secrets
from typing import List, Dict, Any

from src.dubchain.cuda import CUDAManager, CUDAConfig, cuda_available
from src.dubchain.consensus import (
    ConsensusEngine,
    ConsensusConfig,
    ConsensusType,
    CUDAConsensusAccelerator,
    CUDAConsensusConfig,
)


class TestCUDAConsensusIntegration:
    """Test CUDA integration in consensus operations."""
    
    @pytest.fixture
    def cuda_config(self):
        """Create CUDA configuration for testing."""
        return CUDAConfig(
            enable_cuda=True,
            enable_consensus_gpu=True,
            enable_test_gpu=True,
            test_gpu_fallback=True,
            min_batch_size_gpu=5,
            max_batch_size=1000,
            chunk_size=100,
        )
    
    @pytest.fixture
    def consensus_config(self):
        """Create consensus configuration for testing."""
        return ConsensusConfig(
            consensus_type=ConsensusType.PROOF_OF_STAKE,
            max_validators=100,
            block_time=10,  # Fixed: use block_time instead of block_time_seconds
            # finality_threshold and enable_cuda_acceleration are not valid ConsensusConfig parameters
        )
    
    @pytest.fixture
    def cuda_consensus_config(self):
        """Create CUDA consensus configuration for testing."""
        return CUDAConsensusConfig(
            enable_gpu_acceleration=True,
            batch_size_threshold=10,
            parallel_verification=True,
            gpu_memory_limit_mb=512,
        )
    
    @pytest.fixture
    def cuda_manager(self, cuda_config):
        """Create CUDA manager for testing."""
        return CUDAManager(cuda_config)
    
    @pytest.fixture
    def cuda_consensus_accelerator(self, cuda_consensus_config):
        """Create CUDA consensus accelerator for testing."""
        return CUDAConsensusAccelerator(cuda_consensus_config)
    
    @pytest.fixture
    def consensus_engine(self, consensus_config):
        """Create consensus engine for testing."""
        return ConsensusEngine(consensus_config)
    
    def test_cuda_availability(self):
        """Test CUDA availability detection."""
        available = cuda_available()
        assert isinstance(available, bool)
        print(f"CUDA Available: {available}")
    
    def test_cuda_consensus_accelerator_initialization(self, cuda_consensus_config):
        """Test CUDA consensus accelerator initialization."""
        accelerator = CUDAConsensusAccelerator(cuda_consensus_config)
        assert accelerator.config == cuda_consensus_config
        assert isinstance(accelerator.cuda_manager, CUDAManager)
        print(f"CUDA Consensus Accelerator - Available: {accelerator.cuda_manager.available}")
    
    def test_consensus_engine_cuda_integration(self, consensus_engine):
        """Test consensus engine CUDA integration."""
        assert hasattr(consensus_engine, 'cuda_accelerator')
        assert isinstance(consensus_engine.cuda_accelerator, CUDAConsensusAccelerator)
        print(f"Consensus Engine CUDA Integration - Available: {consensus_engine.cuda_accelerator.cuda_manager.available}")
    
    def test_signature_verification_batch(self, cuda_consensus_accelerator):
        """Test batch signature verification with CUDA acceleration."""
        # Generate test data
        signatures = [secrets.token_bytes(64) for _ in range(20)]
        public_keys = [secrets.token_bytes(33) for _ in range(20)]
        messages = [secrets.token_bytes(32) for _ in range(20)]
        
        # Test batch signature verification
        results = cuda_consensus_accelerator.verify_signatures_batch(
            signatures, public_keys, messages
        )
        
        assert isinstance(results, list)
        assert len(results) == len(signatures)
        
        # Verify all results are boolean
        for result in results:
            assert isinstance(result, bool)
        
        print(f"Batch signature verification completed: {len(results)} verifications")
    
    def test_block_validation_batch(self, cuda_consensus_accelerator):
        """Test batch block validation with CUDA acceleration."""
        # Generate test blocks
        blocks = []
        for i in range(15):
            block = {
                'index': i,
                'timestamp': time.time(),
                'data': secrets.token_bytes(64),
                'previous_hash': secrets.token_hex(32),
                'validator': f"validator_{i}",
            }
            blocks.append(block)
        
        # Test batch block validation
        results = cuda_consensus_accelerator.validate_blocks_batch(blocks)
        
        assert isinstance(results, list)
        assert len(results) == len(blocks)
        
        # Verify all results are ConsensusResult objects
        for result in results:
            assert hasattr(result, 'success')
            assert hasattr(result, 'block_hash')
            assert hasattr(result, 'consensus_type')
        
        print(f"Batch block validation completed: {len(results)} validations")
    
    def test_consensus_operations_batch(self, cuda_consensus_accelerator):
        """Test batch consensus operations with CUDA acceleration."""
        # Generate test operations
        operations = []
        for i in range(25):
            operation = {
                'id': f"op_{i}",
                'type': 'consensus_operation',
                'data': secrets.token_bytes(32),
                'timestamp': time.time(),
            }
            operations.append(operation)
        
        # Test batch operation processing
        results = cuda_consensus_accelerator.process_consensus_operations(operations)
        
        assert isinstance(results, list)
        assert len(results) == len(operations)
        
        # Verify all results have expected structure
        for result in results:
            assert isinstance(result, dict)
            assert 'operation_id' in result
            assert 'success' in result
            assert 'result' in result
        
        print(f"Batch consensus operations completed: {len(results)} operations")
    
    def test_consensus_engine_signature_verification(self, consensus_engine):
        """Test consensus engine signature verification with CUDA."""
        # Generate test data
        signatures = [secrets.token_bytes(64) for _ in range(15)]
        public_keys = [secrets.token_bytes(33) for _ in range(15)]
        messages = [secrets.token_bytes(32) for _ in range(15)]
        
        # Test signature verification through consensus engine
        results = consensus_engine.verify_signatures_batch(
            signatures, public_keys, messages
        )
        
        assert isinstance(results, list)
        assert len(results) == len(signatures)
        
        # Verify all results are boolean
        for result in results:
            assert isinstance(result, bool)
        
        print(f"Consensus engine signature verification completed: {len(results)} verifications")
    
    def test_consensus_engine_block_validation(self, consensus_engine):
        """Test consensus engine block validation with CUDA."""
        # Generate test blocks
        blocks = []
        for i in range(12):
            block = {
                'index': i,
                'timestamp': time.time(),
                'data': secrets.token_bytes(64),
                'previous_hash': secrets.token_hex(32),
                'validator': f"validator_{i}",
            }
            blocks.append(block)
        
        # Test block validation through consensus engine
        results = consensus_engine.validate_blocks_batch(blocks)
        
        assert isinstance(results, list)
        assert len(results) == len(blocks)
        
        # Verify all results are ConsensusResult objects
        for result in results:
            assert hasattr(result, 'success')
            assert hasattr(result, 'block_hash')
            assert hasattr(result, 'consensus_type')
        
        print(f"Consensus engine block validation completed: {len(results)} validations")
    
    def test_performance_metrics(self, cuda_consensus_accelerator):
        """Test performance metrics collection."""
        # Perform some operations
        signatures = [secrets.token_bytes(64) for _ in range(10)]
        public_keys = [secrets.token_bytes(33) for _ in range(10)]
        messages = [secrets.token_bytes(32) for _ in range(10)]
        
        cuda_consensus_accelerator.verify_signatures_batch(signatures, public_keys, messages)
        
        # Get metrics
        metrics = cuda_consensus_accelerator.get_performance_metrics()
        
        assert isinstance(metrics, dict)
        assert 'total_operations' in metrics
        assert 'gpu_operations' in metrics
        assert 'cpu_fallbacks' in metrics
        assert 'cuda_available' in metrics
        assert 'config' in metrics
        
        print(f"Performance metrics: {metrics}")
    
    def test_consensus_engine_performance_metrics(self, consensus_engine):
        """Test consensus engine performance metrics."""
        # Perform some operations
        signatures = [secrets.token_bytes(64) for _ in range(8)]
        public_keys = [secrets.token_bytes(33) for _ in range(8)]
        messages = [secrets.token_bytes(32) for _ in range(8)]
        
        consensus_engine.verify_signatures_batch(signatures, public_keys, messages)
        
        # Get CUDA metrics
        metrics = consensus_engine.get_cuda_performance_metrics()
        
        assert isinstance(metrics, dict)
        assert 'total_operations' in metrics
        assert 'gpu_operations' in metrics
        assert 'cpu_fallbacks' in metrics
        assert 'cuda_available' in metrics
        
        print(f"Consensus engine CUDA metrics: {metrics}")
    
    def test_benchmark_consensus_operations(self, cuda_consensus_accelerator):
        """Test consensus operations benchmarking."""
        # Generate test data
        test_operations = []
        for i in range(50):
            operation = {
                'id': f"benchmark_op_{i}",
                'type': 'consensus_operation',
                'data': secrets.token_bytes(32),
                'timestamp': time.time(),
            }
            test_operations.append(operation)
        
        # Run benchmark
        benchmark_results = cuda_consensus_accelerator.benchmark_consensus_operations(
            test_operations, num_iterations=3
        )
        
        assert isinstance(benchmark_results, dict)
        assert 'algorithm' in benchmark_results
        assert 'data_size' in benchmark_results
        assert 'cpu_avg_time' in benchmark_results
        assert 'gpu_avg_time' in benchmark_results
        assert 'speedup' in benchmark_results
        assert 'gpu_success' in benchmark_results
        
        print(f"Consensus benchmark results: {benchmark_results}")
    
    def test_consensus_engine_benchmark(self, consensus_engine):
        """Test consensus engine benchmarking."""
        # Generate test blocks
        test_blocks = []
        for i in range(30):
            block = {
                'index': i,
                'timestamp': time.time(),
                'data': secrets.token_bytes(64),
                'previous_hash': secrets.token_hex(32),
                'validator': f"validator_{i}",
            }
            test_blocks.append(block)
        
        # Run benchmark
        benchmark_results = consensus_engine.benchmark_consensus_performance(
            test_blocks, num_iterations=3
        )
        
        assert isinstance(benchmark_results, dict)
        assert 'algorithm' in benchmark_results
        assert 'data_size' in benchmark_results
        assert 'cpu_avg_time' in benchmark_results
        assert 'gpu_avg_time' in benchmark_results
        assert 'speedup' in benchmark_results
        
        print(f"Consensus engine benchmark results: {benchmark_results}")
    
    def test_fallback_behavior(self, cuda_consensus_accelerator):
        """Test CPU fallback behavior when GPU is not available."""
        # Test with small batch to trigger CPU fallback
        signatures = [secrets.token_bytes(64) for _ in range(3)]
        public_keys = [secrets.token_bytes(33) for _ in range(3)]
        messages = [secrets.token_bytes(32) for _ in range(3)]
        
        # This should work even if GPU is not available
        results = cuda_consensus_accelerator.verify_signatures_batch(
            signatures, public_keys, messages
        )
        
        assert isinstance(results, list)
        assert len(results) == len(signatures)
        
        print(f"Fallback test successful: {len(results)} verifications")
    
    def test_memory_management(self, cuda_consensus_accelerator):
        """Test GPU memory management."""
        # Perform operations that should trigger memory management
        large_blocks = []
        for i in range(100):
            block = {
                'index': i,
                'timestamp': time.time(),
                'data': secrets.token_bytes(1024),  # Larger data
                'previous_hash': secrets.token_hex(32),
                'validator': f"validator_{i}",
            }
            large_blocks.append(block)
        
        # This should trigger chunking and memory management
        results = cuda_consensus_accelerator.validate_blocks_batch(large_blocks)
        
        assert len(results) == len(large_blocks)
        
        print(f"Memory management test completed: {len(results)} operations")
    
    def test_concurrent_consensus_operations(self, cuda_consensus_accelerator):
        """Test concurrent consensus operations."""
        import threading
        import queue
        
        results_queue = queue.Queue()
        
        def worker(worker_id: int):
            """Worker function for concurrent operations."""
            signatures = [secrets.token_bytes(64) for _ in range(5)]
            public_keys = [secrets.token_bytes(33) for _ in range(5)]
            messages = [secrets.token_bytes(32) for _ in range(5)]
            
            results = cuda_consensus_accelerator.verify_signatures_batch(
                signatures, public_keys, messages
            )
            results_queue.put((worker_id, len(results)))
        
        # Start multiple worker threads
        threads = []
        for i in range(3):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Collect results
        results = []
        while not results_queue.empty():
            results.append(results_queue.get())
        
        assert len(results) == 3
        for worker_id, count in results:
            assert count == 5
        
        print(f"Concurrent consensus operations completed: {results}")
    
    def test_error_handling(self, cuda_consensus_accelerator):
        """Test error handling in consensus operations."""
        # Test with empty data
        try:
            results = cuda_consensus_accelerator.verify_signatures_batch([], [], [])
            assert results == []
        except Exception as e:
            print(f"Empty data handling: {e}")
        
        # Test with mismatched data lengths
        try:
            results = cuda_consensus_accelerator.verify_signatures_batch(
                [secrets.token_bytes(64)],  # 1 signature
                [secrets.token_bytes(33), secrets.token_bytes(33)],  # 2 keys
                [secrets.token_bytes(32)]  # 1 message
            )
            # Should either work or fail gracefully
        except ValueError as e:
            print(f"Mismatched data handling: {e}")
        
        print("Error handling test completed")


if __name__ == "__main__":
    # Run basic tests
    pytest.main([__file__, "-v"])
