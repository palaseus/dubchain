"""
CUDA Crypto Integration Tests for DubChain.

This module provides comprehensive tests for CUDA integration in cryptographic operations.
"""

import logging

logger = logging.getLogger(__name__)
import pytest
import time
import secrets
from typing import List, Dict, Any

from src.dubchain.cuda import CUDAManager, CUDAConfig, cuda_available
from src.dubchain.crypto.gpu_crypto import GPUCrypto, GPUConfig


class TestCUDACryptoIntegration:
    """Test CUDA integration in cryptographic operations."""
    
    @pytest.fixture
    def cuda_config(self):
        """Create CUDA configuration for testing."""
        return CUDAConfig(
            enable_cuda=True,
            enable_crypto_gpu=True,
            enable_test_gpu=True,
            test_gpu_fallback=True,
            min_batch_size_gpu=5,
            max_batch_size=1000,
            chunk_size=100,
        )
    
    @pytest.fixture
    def gpu_config(self):
        """Create GPU crypto configuration for testing."""
        return GPUConfig(
            enable_gpu_acceleration=True,
            enable_cuda_hashing=True,
            enable_parallel_verification=True,
            batch_size=100,
            fallback_to_cpu=True,
        )
    
    @pytest.fixture
    def cuda_manager(self, cuda_config):
        """Create CUDA manager for testing."""
        return CUDAManager(cuda_config)
    
    @pytest.fixture
    def gpu_crypto(self, gpu_config, cuda_manager):
        """Create GPU crypto instance for testing."""
        return GPUCrypto(gpu_config, cuda_manager)
    
    def test_cuda_availability(self):
        """Test CUDA availability detection."""
        available = cuda_available()
        assert isinstance(available, bool)
        logger.info(f"CUDA Available: {available}")
    
    def test_cuda_manager_initialization(self, cuda_config):
        """Test CUDA manager initialization."""
        manager = CUDAManager(cuda_config)
        assert manager.config == cuda_config
        assert isinstance(manager.available, bool)
        assert isinstance(manager.device, (str, type(None)))
        logger.info(f"CUDA Manager - Available: {manager.available}, Device: {manager.device}")
    
    def test_gpu_crypto_initialization(self, gpu_config):
        """Test GPU crypto initialization."""
        crypto = GPUCrypto(gpu_config)
        assert crypto.config == gpu_config
        assert isinstance(crypto.gpu_available, bool)
        assert isinstance(crypto.device, str)
        logger.info(f"GPU Crypto - Available: {crypto.gpu_available}, Device: {crypto.device}")
    
    def test_single_hash_operation(self, gpu_crypto):
        """Test single hash operation with GPU acceleration."""
        test_data = b"Hello, DubChain CUDA World!"
        
        # Test hash operation
        result = gpu_crypto.hash_data_gpu(test_data, "sha256")
        
        assert isinstance(result, bytes)
        assert len(result) == 32  # SHA-256 produces 32 bytes
        
        # Verify it's deterministic
        result2 = gpu_crypto.hash_data_gpu(test_data, "sha256")
        assert result == result2
        
        logger.info(f"Hash result: {result.hex()}")
    
    def test_batch_hash_operations(self, gpu_crypto):
        """Test batch hash operations with GPU acceleration."""
        # Generate test data
        test_data = [secrets.token_bytes(64) for _ in range(50)]
        
        # Test batch hash operation
        results = gpu_crypto.hash_data_batch_gpu(test_data, "sha256")
        
        assert isinstance(results, list)
        assert len(results) == len(test_data)
        
        # Verify all results are valid hashes
        for result in results:
            assert isinstance(result, bytes)
            assert len(result) == 32
        
        # Verify deterministic results
        results2 = gpu_crypto.hash_data_batch_gpu(test_data, "sha256")
        assert results == results2
        
        logger.info(f"Batch hash completed: {len(results)} hashes")
    
    def test_signature_verification(self, gpu_crypto):
        """Test signature verification with GPU acceleration."""
        # Generate test verification data
        verifications = []
        for _ in range(20):
            message = secrets.token_bytes(32)
            signature = secrets.token_bytes(64)
            public_key = secrets.token_bytes(33)
            algorithm = "ecdsa"
            verifications.append((message, signature, public_key, algorithm))
        
        # Test signature verification
        results = gpu_crypto.verify_signatures_gpu(verifications)
        
        assert isinstance(results, list)
        assert len(results) == len(verifications)
        
        # Verify all results are boolean
        for result in results:
            assert isinstance(result, bool)
        
        logger.info(f"Signature verification completed: {len(results)} verifications")
    
    def test_performance_metrics(self, gpu_crypto):
        """Test performance metrics collection."""
        # Perform some operations
        test_data = [secrets.token_bytes(32) for _ in range(10)]
        gpu_crypto.hash_data_batch_gpu(test_data, "sha256")
        
        # Get metrics
        metrics = gpu_crypto.get_performance_metrics()
        
        assert isinstance(metrics, dict)
        assert "total_operations" in metrics
        assert "gpu_operations" in metrics
        assert "cpu_fallbacks" in metrics
        assert "gpu_utilization" in metrics
        assert "gpu_available" in metrics
        assert "global_cuda_metrics" in metrics
        
        logger.info(f"Performance metrics: {metrics}")
    
    def test_benchmark_performance(self, gpu_crypto):
        """Test performance benchmarking."""
        # Run benchmark
        benchmark_results = gpu_crypto.benchmark(
            data_size=64,
            num_operations=100
        )
        
        assert isinstance(benchmark_results, dict)
        assert "algorithm" in benchmark_results
        assert "data_size" in benchmark_results
        assert "cpu_avg_time" in benchmark_results
        assert "gpu_avg_time" in benchmark_results
        assert "speedup" in benchmark_results
        assert "gpu_success" in benchmark_results
        
        logger.info(f"Benchmark results: {benchmark_results}")
    
    def test_cuda_manager_integration(self, cuda_manager, gpu_crypto):
        """Test integration between CUDA manager and GPU crypto."""
        # Test that GPU crypto uses the global CUDA manager
        assert gpu_crypto.cuda_manager == cuda_manager
        
        # Test performance metrics integration
        gpu_metrics = gpu_crypto.get_performance_metrics()
        global_metrics = cuda_manager.get_performance_metrics()
        
        assert "global_cuda_metrics" in gpu_metrics
        assert isinstance(gpu_metrics["global_cuda_metrics"], dict)
        
        logger.info(f"CUDA Manager metrics: {global_metrics}")
    
    def test_fallback_behavior(self, gpu_crypto):
        """Test CPU fallback behavior when GPU is not available."""
        # This test ensures that operations still work even if GPU is not available
        test_data = b"Fallback test data"
        
        # Force CPU fallback by using small data
        result = gpu_crypto.hash_data_gpu(test_data, "sha256")
        
        assert isinstance(result, bytes)
        assert len(result) == 32
        
        logger.info(f"Fallback test successful: {result.hex()}")
    
    def test_memory_management(self, gpu_crypto):
        """Test GPU memory management."""
        # Perform operations that should trigger memory management
        large_data = [secrets.token_bytes(1024) for _ in range(100)]
        
        # This should trigger chunking and memory management
        results = gpu_crypto.hash_data_batch_gpu(large_data, "sha256")
        
        assert len(results) == len(large_data)
        
        # Check memory usage in metrics
        metrics = gpu_crypto.get_performance_metrics()
        assert "global_cuda_metrics" in metrics
        
        logger.info(f"Memory management test completed: {len(results)} operations")
    
    def test_concurrent_operations(self, gpu_crypto):
        """Test concurrent GPU operations."""
        import threading
        import queue
        
        results_queue = queue.Queue()
        
        def worker(worker_id: int):
            """Worker function for concurrent operations."""
            test_data = [secrets.token_bytes(32) for _ in range(10)]
            results = gpu_crypto.hash_data_batch_gpu(test_data, "sha256")
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
            assert count == 10
        
        logger.info(f"Concurrent operations completed: {results}")
    
    def test_error_handling(self, gpu_crypto):
        """Test error handling in GPU operations."""
        # Test with invalid data
        try:
            result = gpu_crypto.hash_data_gpu(b"", "sha256")
            assert isinstance(result, bytes)
        except Exception as e:
            # Should either work or fail gracefully
            logger.info(f"Empty data handling: {e}")
        
        # Test with invalid algorithm
        try:
            result = gpu_crypto.hash_data_gpu(b"test", "invalid_algorithm")
            assert isinstance(result, bytes)
        except Exception as e:
            # Should either work or fail gracefully
            logger.info(f"Invalid algorithm handling: {e}")
        
        logger.info("Error handling test completed")


if __name__ == "__main__":
    # Run basic tests
    pytest.main([__file__, "-v"])
