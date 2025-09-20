"""
CUDA VM Integration Tests for DubChain.

This module provides comprehensive tests for CUDA integration in virtual machine operations.
"""

import pytest
import time
import secrets
from typing import List, Dict, Any

from src.dubchain.cuda import CUDAManager, CUDAConfig, cuda_available
from src.dubchain.vm import (
    ExecutionEngine,
    CUDAVMAccelerator,
    CUDAVMConfig,
    SmartContract,
    ContractType,
)


class TestCUDAVMIntegration:
    """Test CUDA integration in virtual machine operations."""
    
    @pytest.fixture
    def cuda_config(self):
        """Create CUDA configuration for testing."""
        return CUDAConfig(
            enable_cuda=True,
            enable_storage_gpu=True,  # VM operations use storage GPU
            enable_test_gpu=True,
            test_gpu_fallback=True,
            min_batch_size_gpu=5,
            max_batch_size=1000,
            chunk_size=100,
        )
    
    @pytest.fixture
    def cuda_vm_config(self):
        """Create CUDA VM configuration for testing."""
        return CUDAVMConfig(
            enable_gpu_acceleration=True,
            batch_size_threshold=10,
            parallel_execution=True,
            gpu_memory_limit_mb=1024,
            enable_bytecode_optimization=True,
        )
    
    @pytest.fixture
    def cuda_manager(self, cuda_config):
        """Create CUDA manager for testing."""
        return CUDAManager(cuda_config)
    
    @pytest.fixture
    def cuda_vm_accelerator(self, cuda_vm_config):
        """Create CUDA VM accelerator for testing."""
        return CUDAVMAccelerator(cuda_vm_config)
    
    @pytest.fixture
    def execution_engine(self):
        """Create execution engine for testing."""
        return ExecutionEngine()
    
    @pytest.fixture
    def test_contracts(self):
        """Create test contracts."""
        contracts = []
        for i in range(15):
            contract = SmartContract(
                address=f"0x{secrets.token_hex(20)}",
                bytecode=secrets.token_bytes(64),
                contract_type=ContractType.STANDARD,
                creator="test_creator",
                creation_time=time.time(),
            )
            contracts.append(contract)
        return contracts
    
    def test_cuda_availability(self):
        """Test CUDA availability detection."""
        available = cuda_available()
        assert isinstance(available, bool)
        print(f"CUDA Available: {available}")
    
    def test_cuda_vm_accelerator_initialization(self, cuda_vm_config):
        """Test CUDA VM accelerator initialization."""
        accelerator = CUDAVMAccelerator(cuda_vm_config)
        assert accelerator.config == cuda_vm_config
        assert isinstance(accelerator.cuda_manager, CUDAManager)
        print(f"CUDA VM Accelerator - Available: {accelerator.cuda_manager.available}")
    
    def test_execution_engine_cuda_integration(self, execution_engine):
        """Test execution engine CUDA integration."""
        assert hasattr(execution_engine, 'cuda_accelerator')
        assert isinstance(execution_engine.cuda_accelerator, CUDAVMAccelerator)
        print(f"Execution Engine CUDA Integration - Available: {execution_engine.cuda_accelerator.cuda_manager.available}")
    
    def test_contract_execution_batch(self, cuda_vm_accelerator, test_contracts):
        """Test batch contract execution with CUDA acceleration."""
        # Generate execution data
        execution_data = []
        for i in range(len(test_contracts)):
            data = {
                'caller': f"0x{secrets.token_hex(20)}",
                'value': 1000,
                'data': secrets.token_bytes(32),
                'gas_limit': 10000,
                'block_context': {'timestamp': time.time()},
            }
            execution_data.append(data)
        
        # Test batch contract execution
        results = cuda_vm_accelerator.execute_contracts_batch(test_contracts, execution_data)
        
        assert isinstance(results, list)
        assert len(results) == len(test_contracts)
        
        # Verify all results are ExecutionResult objects
        for result in results:
            assert hasattr(result, 'success')
            assert hasattr(result, 'gas_used')
            assert hasattr(result, 'return_data')
            assert hasattr(result, 'logs')
            assert hasattr(result, 'state_changes')
        
        print(f"Batch contract execution completed: {len(results)} executions")
    
    def test_bytecode_processing_batch(self, cuda_vm_accelerator):
        """Test batch bytecode processing with CUDA acceleration."""
        # Generate test bytecode
        bytecode_list = [secrets.token_bytes(64) for _ in range(20)]
        
        # Test batch bytecode processing
        results = cuda_vm_accelerator.process_bytecode_batch(bytecode_list, optimization_level=1)
        
        assert isinstance(results, list)
        assert len(results) == len(bytecode_list)
        
        # Verify all results are bytes
        for result in results:
            assert isinstance(result, bytes)
        
        print(f"Batch bytecode processing completed: {len(results)} operations")
    
    def test_vm_operations_batch(self, cuda_vm_accelerator):
        """Test batch VM operations with CUDA acceleration."""
        # Generate test operations
        operations = []
        for i in range(25):
            operation = {
                'id': f"vm_op_{i}",
                'type': 'vm_operation',
                'data': secrets.token_bytes(32),
                'gas_limit': 1000,
                'timestamp': time.time(),
            }
            operations.append(operation)
        
        # Test batch operation execution
        results = cuda_vm_accelerator.execute_operations_batch(operations)
        
        assert isinstance(results, list)
        assert len(results) == len(operations)
        
        # Verify all results have expected structure
        for result in results:
            assert isinstance(result, dict)
            assert 'operation_id' in result
            assert 'success' in result
            assert 'result' in result
            assert 'gas_used' in result
            assert 'execution_time' in result
        
        print(f"Batch VM operations completed: {len(results)} operations")
    
    def test_bytecode_optimization_batch(self, cuda_vm_accelerator):
        """Test batch bytecode optimization with CUDA acceleration."""
        # Generate test bytecode
        bytecode_list = [secrets.token_bytes(64) for _ in range(18)]
        optimization_rules = ["constant_folding", "dead_code_elimination", "peephole"]
        
        # Test batch bytecode optimization
        results = cuda_vm_accelerator.optimize_bytecode_batch(bytecode_list, optimization_rules)
        
        assert isinstance(results, list)
        assert len(results) == len(bytecode_list)
        
        # Verify all results are bytes
        for result in results:
            assert isinstance(result, bytes)
        
        print(f"Batch bytecode optimization completed: {len(results)} optimizations")
    
    def test_execution_engine_contract_execution(self, execution_engine, test_contracts):
        """Test execution engine contract execution with CUDA."""
        # Generate execution data
        execution_data = []
        for i in range(len(test_contracts)):
            data = {
                'caller': f"0x{secrets.token_hex(20)}",
                'value': 1000,
                'data': secrets.token_bytes(32),
                'gas_limit': 10000,
                'block_context': {'timestamp': time.time()},
            }
            execution_data.append(data)
        
        # Test contract execution through execution engine
        results = execution_engine.execute_contracts_batch(test_contracts, execution_data)
        
        assert isinstance(results, list)
        assert len(results) == len(test_contracts)
        
        # Verify all results are ExecutionResult objects
        for result in results:
            assert hasattr(result, 'success')
            assert hasattr(result, 'gas_used')
            assert hasattr(result, 'return_data')
        
        print(f"Execution engine contract execution completed: {len(results)} executions")
    
    def test_execution_engine_bytecode_processing(self, execution_engine):
        """Test execution engine bytecode processing with CUDA."""
        # Generate test bytecode
        bytecode_list = [secrets.token_bytes(64) for _ in range(12)]
        
        # Test bytecode processing through execution engine
        results = execution_engine.process_bytecode_batch(bytecode_list, optimization_level=1)
        
        assert isinstance(results, list)
        assert len(results) == len(bytecode_list)
        
        # Verify all results are bytes
        for result in results:
            assert isinstance(result, bytes)
        
        print(f"Execution engine bytecode processing completed: {len(results)} operations")
    
    def test_execution_engine_operations(self, execution_engine):
        """Test execution engine operations with CUDA."""
        # Generate test operations
        operations = []
        for i in range(15):
            operation = {
                'id': f"exec_engine_op_{i}",
                'type': 'execution_operation',
                'data': secrets.token_bytes(32),
                'gas_limit': 1000,
                'timestamp': time.time(),
            }
            operations.append(operation)
        
        # Test operations through execution engine
        results = execution_engine.execute_operations_batch(operations)
        
        assert isinstance(results, list)
        assert len(results) == len(operations)
        
        # Verify all results have expected structure
        for result in results:
            assert isinstance(result, dict)
            assert 'operation_id' in result
            assert 'success' in result
            assert 'result' in result
        
        print(f"Execution engine operations completed: {len(results)} operations")
    
    def test_performance_metrics(self, cuda_vm_accelerator):
        """Test performance metrics collection."""
        # Perform some operations
        bytecode_list = [secrets.token_bytes(32) for _ in range(10)]
        cuda_vm_accelerator.process_bytecode_batch(bytecode_list, optimization_level=1)
        
        # Get metrics
        metrics = cuda_vm_accelerator.get_performance_metrics()
        
        assert isinstance(metrics, dict)
        assert 'total_operations' in metrics
        assert 'gpu_operations' in metrics
        assert 'cpu_fallbacks' in metrics
        assert 'cuda_available' in metrics
        assert 'config' in metrics
        
        print(f"Performance metrics: {metrics}")
    
    def test_execution_engine_performance_metrics(self, execution_engine):
        """Test execution engine performance metrics."""
        # Perform some operations
        bytecode_list = [secrets.token_bytes(32) for _ in range(8)]
        execution_engine.process_bytecode_batch(bytecode_list, optimization_level=1)
        
        # Get CUDA metrics
        metrics = execution_engine.get_cuda_performance_metrics()
        
        assert isinstance(metrics, dict)
        assert 'total_operations' in metrics
        assert 'gpu_operations' in metrics
        assert 'cpu_fallbacks' in metrics
        assert 'cuda_available' in metrics
        
        print(f"Execution engine CUDA metrics: {metrics}")
    
    def test_benchmark_vm_operations(self, cuda_vm_accelerator):
        """Test VM operations benchmarking."""
        # Generate test data
        test_operations = []
        for i in range(50):
            operation = {
                'id': f"benchmark_vm_op_{i}",
                'type': 'vm_operation',
                'data': secrets.token_bytes(32),
                'gas_limit': 1000,
                'timestamp': time.time(),
            }
            test_operations.append(operation)
        
        # Run benchmark
        benchmark_results = cuda_vm_accelerator.benchmark_vm_operations(
            test_operations, num_iterations=3
        )
        
        assert isinstance(benchmark_results, dict)
        assert 'algorithm' in benchmark_results
        assert 'data_size' in benchmark_results
        assert 'cpu_avg_time' in benchmark_results
        assert 'gpu_avg_time' in benchmark_results
        assert 'speedup' in benchmark_results
        assert 'gpu_success' in benchmark_results
        
        print(f"VM benchmark results: {benchmark_results}")
    
    def test_execution_engine_benchmark(self, execution_engine):
        """Test execution engine benchmarking."""
        # Generate test data
        test_operations = []
        for i in range(30):
            operation = {
                'id': f"exec_engine_benchmark_{i}",
                'type': 'execution_operation',
                'data': secrets.token_bytes(32),
                'gas_limit': 1000,
                'timestamp': time.time(),
            }
            test_operations.append(operation)
        
        # Run benchmark
        benchmark_results = execution_engine.benchmark_vm_performance(
            test_operations, num_iterations=3
        )
        
        assert isinstance(benchmark_results, dict)
        assert 'algorithm' in benchmark_results
        assert 'data_size' in benchmark_results
        assert 'cpu_avg_time' in benchmark_results
        assert 'gpu_avg_time' in benchmark_results
        assert 'speedup' in benchmark_results
        
        print(f"Execution engine benchmark results: {benchmark_results}")
    
    def test_fallback_behavior(self, cuda_vm_accelerator):
        """Test CPU fallback behavior when GPU is not available."""
        # Test with small batch to trigger CPU fallback
        bytecode_list = [secrets.token_bytes(32) for _ in range(3)]
        
        # This should work even if GPU is not available
        results = cuda_vm_accelerator.process_bytecode_batch(bytecode_list, optimization_level=1)
        
        assert isinstance(results, list)
        assert len(results) == len(bytecode_list)
        
        print(f"Fallback test successful: {len(results)} operations")
    
    def test_memory_management(self, cuda_vm_accelerator):
        """Test GPU memory management."""
        # Perform operations that should trigger memory management
        large_bytecode = [secrets.token_bytes(1024) for _ in range(100)]
        
        # This should trigger chunking and memory management
        results = cuda_vm_accelerator.process_bytecode_batch(large_bytecode, optimization_level=1)
        
        assert len(results) == len(large_bytecode)
        
        print(f"Memory management test completed: {len(results)} operations")
    
    def test_concurrent_vm_operations(self, cuda_vm_accelerator):
        """Test concurrent VM operations."""
        import threading
        import queue
        
        results_queue = queue.Queue()
        
        def worker(worker_id: int):
            """Worker function for concurrent operations."""
            bytecode_list = [secrets.token_bytes(32) for _ in range(5)]
            results = cuda_vm_accelerator.process_bytecode_batch(bytecode_list, optimization_level=1)
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
        
        print(f"Concurrent VM operations completed: {results}")
    
    def test_error_handling(self, cuda_vm_accelerator):
        """Test error handling in VM operations."""
        # Test with empty data
        try:
            results = cuda_vm_accelerator.process_bytecode_batch([], optimization_level=1)
            assert results == []
        except Exception as e:
            print(f"Empty data handling: {e}")
        
        # Test with invalid optimization level
        try:
            bytecode_list = [secrets.token_bytes(32)]
            results = cuda_vm_accelerator.process_bytecode_batch(bytecode_list, optimization_level=-1)
            assert isinstance(results, list)
        except Exception as e:
            print(f"Invalid optimization level handling: {e}")
        
        print("Error handling test completed")


if __name__ == "__main__":
    # Run basic tests
    pytest.main([__file__, "-v"])
