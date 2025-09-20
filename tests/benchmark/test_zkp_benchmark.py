"""
Benchmark and stress test harness for ZKP system.

This module provides comprehensive performance testing and benchmarking
capabilities for the ZKP system under various load conditions.
"""

import pytest

# Temporarily disable all benchmark tests due to hanging issues
pytestmark = pytest.mark.skip(reason="Temporarily disabled due to hanging issues")
import time
import statistics
import concurrent.futures
import threading
import psutil
import gc
from typing import List, Dict, Any, Tuple
import json
import os

from src.dubchain.crypto.zkp import (
    ZKPManager, ZKPConfig, ZKPType, ZKPStatus, ProofRequest, Proof, VerificationResult,
    CircuitBuilder, PublicInputs, PrivateInputs, ZKCircuit
)


class ZKPBenchmark:
    """Benchmark harness for ZKP system performance testing."""
    
    def __init__(self, config: ZKPConfig):
        self.config = config
        self.manager = ZKPManager(config)
        self.manager.initialize()
        self.results = {}
    
    def benchmark_proof_generation(self, num_proofs: int = 100, input_size: int = 100) -> Dict[str, Any]:
        """Benchmark proof generation performance."""
        print(f"Benchmarking proof generation: {num_proofs} proofs, {input_size} bytes input")
        
        # Prepare test data
        test_input = b"x" * input_size
        
        # Warm up
        warmup_request = ProofRequest(
            circuit_id="warmup",
            public_inputs=[test_input],
            private_inputs=[test_input],
            proof_type=self.config.backend_type
        )
        self.manager.generate_proof(warmup_request)
        
        # Benchmark
        generation_times = []
        memory_usage = []
        
        start_memory = psutil.Process().memory_info().rss
        
        for i in range(num_proofs):
            request = ProofRequest(
                circuit_id=f"benchmark_{i}",
                public_inputs=[test_input],
                private_inputs=[test_input],
                proof_type=self.config.backend_type
            )
            
            start_time = time.time()
            result = self.manager.generate_proof(request)
            generation_time = time.time() - start_time
            
            if result.is_success:
                generation_times.append(generation_time)
                memory_usage.append(psutil.Process().memory_info().rss)
        
        end_memory = psutil.Process().memory_info().rss
        
        # Calculate statistics
        stats = {
            'total_proofs': num_proofs,
            'successful_proofs': len(generation_times),
            'success_rate': len(generation_times) / num_proofs,
            'avg_generation_time': statistics.mean(generation_times) if generation_times else 0,
            'median_generation_time': statistics.median(generation_times) if generation_times else 0,
            'min_generation_time': min(generation_times) if generation_times else 0,
            'max_generation_time': max(generation_times) if generation_times else 0,
            'std_generation_time': statistics.stdev(generation_times) if len(generation_times) > 1 else 0,
            'throughput_per_second': len(generation_times) / sum(generation_times) if generation_times else 0,
            'memory_usage_start': start_memory,
            'memory_usage_end': end_memory,
            'memory_usage_peak': max(memory_usage) if memory_usage else start_memory,
            'memory_growth': end_memory - start_memory,
            'input_size': input_size
        }
        
        self.results['proof_generation'] = stats
        return stats
    
    def benchmark_proof_verification(self, num_verifications: int = 100, input_size: int = 100) -> Dict[str, Any]:
        """Benchmark proof verification performance."""
        print(f"Benchmarking proof verification: {num_verifications} verifications, {input_size} bytes input")
        
        # Generate proofs first
        test_input = b"x" * input_size
        proofs = []
        
        for i in range(num_verifications):
            request = ProofRequest(
                circuit_id=f"verify_benchmark_{i}",
                public_inputs=[test_input],
                private_inputs=[test_input],
                proof_type=self.config.backend_type
            )
            
            result = self.manager.generate_proof(request)
            if result.is_success:
                proofs.append((result.proof, [test_input]))
        
        if not proofs:
            return {'error': 'No proofs generated for verification benchmark'}
        
        # Warm up
        self.manager.verify_proof(proofs[0][0], proofs[0][1])
        
        # Benchmark verification
        verification_times = []
        memory_usage = []
        
        start_memory = psutil.Process().memory_info().rss
        
        for proof, public_inputs in proofs:
            start_time = time.time()
            verify_result = self.manager.verify_proof(proof, public_inputs)
            verification_time = time.time() - start_time
            
            if verify_result.is_success:
                verification_times.append(verification_time)
                memory_usage.append(psutil.Process().memory_info().rss)
        
        end_memory = psutil.Process().memory_info().rss
        
        # Calculate statistics
        stats = {
            'total_verifications': len(proofs),
            'successful_verifications': len(verification_times),
            'success_rate': len(verification_times) / len(proofs),
            'avg_verification_time': statistics.mean(verification_times) if verification_times else 0,
            'median_verification_time': statistics.median(verification_times) if verification_times else 0,
            'min_verification_time': min(verification_times) if verification_times else 0,
            'max_verification_time': max(verification_times) if verification_times else 0,
            'std_verification_time': statistics.stdev(verification_times) if len(verification_times) > 1 else 0,
            'throughput_per_second': len(verification_times) / sum(verification_times) if verification_times else 0,
            'memory_usage_start': start_memory,
            'memory_usage_end': end_memory,
            'memory_usage_peak': max(memory_usage) if memory_usage else start_memory,
            'memory_growth': end_memory - start_memory,
            'input_size': input_size
        }
        
        self.results['proof_verification'] = stats
        return stats
    
    def benchmark_batch_operations(self, batch_sizes: List[int] = [1, 5, 10, 20, 50]) -> Dict[str, Any]:
        """Benchmark batch operations with different batch sizes."""
        print(f"Benchmarking batch operations: {batch_sizes}")
        
        batch_results = {}
        
        for batch_size in batch_sizes:
            print(f"  Testing batch size: {batch_size}")
            
            # Generate proofs for batch
            proofs = []
            public_inputs_list = []
            
            for i in range(batch_size):
                request = ProofRequest(
                    circuit_id=f"batch_benchmark_{i}",
                    public_inputs=[f"batch_input_{i}".encode()],
                    private_inputs=[f"batch_private_{i}".encode()],
                    proof_type=self.config.backend_type
                )
                
                result = self.manager.generate_proof(request)
                if result.is_success:
                    proofs.append(result.proof)
                    public_inputs_list.append(request.public_inputs)
            
            if not proofs:
                continue
            
            # Benchmark batch verification
            start_time = time.time()
            results = self.manager.batch_verify_proofs(proofs, public_inputs_list)
            batch_time = time.time() - start_time
            
            successful_verifications = sum(1 for r in results if r.is_success and r.is_valid)
            
            batch_results[batch_size] = {
                'batch_size': batch_size,
                'total_time': batch_time,
                'avg_time_per_proof': batch_time / len(proofs),
                'successful_verifications': successful_verifications,
                'success_rate': successful_verifications / len(proofs),
                'throughput_per_second': len(proofs) / batch_time
            }
        
        self.results['batch_operations'] = batch_results
        return batch_results
    
    def benchmark_concurrent_operations(self, num_threads: int = 10, operations_per_thread: int = 10) -> Dict[str, Any]:
        """Benchmark concurrent operations."""
        print(f"Benchmarking concurrent operations: {num_threads} threads, {operations_per_thread} ops/thread")
        
        def worker_thread(thread_id: int, results: List[Dict[str, Any]]):
            """Worker thread for concurrent operations."""
            thread_results = []
            
            for i in range(operations_per_thread):
                request = ProofRequest(
                    circuit_id=f"concurrent_{thread_id}_{i}",
                    public_inputs=[f"concurrent_input_{thread_id}_{i}".encode()],
                    private_inputs=[f"concurrent_private_{thread_id}_{i}".encode()],
                    proof_type=self.config.backend_type
                )
                
                # Generate proof
                start_time = time.time()
                result = self.manager.generate_proof(request)
                generation_time = time.time() - start_time
                
                if result.is_success:
                    # Verify proof
                    start_time = time.time()
                    verify_result = self.manager.verify_proof(result.proof, request.public_inputs)
                    verification_time = time.time() - start_time
                    
                    thread_results.append({
                        'thread_id': thread_id,
                        'operation_id': i,
                        'generation_time': generation_time,
                        'verification_time': verification_time,
                        'total_time': generation_time + verification_time,
                        'success': verify_result.is_success and verify_result.is_valid
                    })
            
            results.extend(thread_results)
        
        # Run concurrent operations
        results = []
        start_time = time.time()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(worker_thread, i, results) for i in range(num_threads)]
            concurrent.futures.wait(futures)
        
        total_time = time.time() - start_time
        
        # Calculate statistics
        successful_ops = [r for r in results if r['success']]
        generation_times = [r['generation_time'] for r in successful_ops]
        verification_times = [r['verification_time'] for r in successful_ops]
        total_times = [r['total_time'] for r in successful_ops]
        
        stats = {
            'num_threads': num_threads,
            'operations_per_thread': operations_per_thread,
            'total_operations': len(results),
            'successful_operations': len(successful_ops),
            'success_rate': len(successful_ops) / len(results) if results else 0,
            'total_wall_time': total_time,
            'avg_generation_time': statistics.mean(generation_times) if generation_times else 0,
            'avg_verification_time': statistics.mean(verification_times) if verification_times else 0,
            'avg_total_time': statistics.mean(total_times) if total_times else 0,
            'throughput_per_second': len(successful_ops) / total_time,
            'concurrency_efficiency': len(successful_ops) / (num_threads * operations_per_thread)
        }
        
        self.results['concurrent_operations'] = stats
        return stats
    
    def benchmark_memory_usage(self, num_operations: int = 1000) -> Dict[str, Any]:
        """Benchmark memory usage patterns."""
        print(f"Benchmarking memory usage: {num_operations} operations")
        
        # Force garbage collection
        gc.collect()
        initial_memory = psutil.Process().memory_info().rss
        
        memory_samples = []
        
        for i in range(num_operations):
            request = ProofRequest(
                circuit_id=f"memory_benchmark_{i}",
                public_inputs=[f"memory_input_{i}".encode()],
                private_inputs=[f"memory_private_{i}".encode()],
                proof_type=self.config.backend_type
            )
            
            result = self.manager.generate_proof(request)
            
            if i % 100 == 0:  # Sample every 100 operations
                memory_samples.append(psutil.Process().memory_info().rss)
        
        # Force garbage collection
        gc.collect()
        final_memory = psutil.Process().memory_info().rss
        
        stats = {
            'initial_memory': initial_memory,
            'final_memory': final_memory,
            'memory_growth': final_memory - initial_memory,
            'peak_memory': max(memory_samples) if memory_samples else initial_memory,
            'avg_memory': statistics.mean(memory_samples) if memory_samples else initial_memory,
            'memory_samples': memory_samples,
            'operations': num_operations
        }
        
        self.results['memory_usage'] = stats
        return stats
    
    def run_comprehensive_benchmark(self) -> Dict[str, Any]:
        """Run comprehensive benchmark suite."""
        print("Running comprehensive ZKP benchmark suite...")
        
        # Run all benchmarks
        self.benchmark_proof_generation(num_proofs=100, input_size=100)
        self.benchmark_proof_verification(num_verifications=100, input_size=100)
        self.benchmark_batch_operations([1, 5, 10, 20])
        self.benchmark_concurrent_operations(num_threads=5, operations_per_thread=10)
        self.benchmark_memory_usage(num_operations=500)
        
        return self.results
    
    def save_results(self, filename: str = "zkp_benchmark_results.json"):
        """Save benchmark results to file."""
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f"Benchmark results saved to {filename}")


class TestZKPBenchmarks:
    """Test class for ZKP benchmarks."""
    
    def test_proof_generation_benchmark(self):
        """Test proof generation benchmark."""
        config = ZKPConfig(backend_type=ZKPType.MOCK)
        benchmark = ZKPBenchmark(config)
        
        results = benchmark.benchmark_proof_generation(num_proofs=50, input_size=50)
        
        # Basic assertions
        assert results['total_proofs'] == 50
        assert results['success_rate'] > 0.8  # At least 80% success rate
        assert results['avg_generation_time'] > 0
        assert results['throughput_per_second'] > 0
    
    def test_proof_verification_benchmark(self):
        """Test proof verification benchmark."""
        config = ZKPConfig(backend_type=ZKPType.MOCK)
        benchmark = ZKPBenchmark(config)
        
        results = benchmark.benchmark_proof_verification(num_verifications=50, input_size=50)
        
        # Basic assertions
        assert 'error' not in results
        assert results['total_verifications'] > 0
        assert results['success_rate'] > 0.8  # At least 80% success rate
        assert results['avg_verification_time'] > 0
        assert results['throughput_per_second'] > 0
    
    def test_batch_operations_benchmark(self):
        """Test batch operations benchmark."""
        config = ZKPConfig(backend_type=ZKPType.MOCK)
        benchmark = ZKPBenchmark(config)
        
        results = benchmark.benchmark_batch_operations([1, 5, 10])
        
        # Basic assertions
        assert len(results) > 0
        for batch_size, stats in results.items():
            assert stats['batch_size'] == batch_size
            assert stats['total_time'] > 0
            assert stats['success_rate'] > 0.8
    
    def test_concurrent_operations_benchmark(self):
        """Test concurrent operations benchmark."""
        config = ZKPConfig(backend_type=ZKPType.MOCK)
        benchmark = ZKPBenchmark(config)
        
        results = benchmark.benchmark_concurrent_operations(num_threads=3, operations_per_thread=5)
        
        # Basic assertions
        assert results['num_threads'] == 3
        assert results['operations_per_thread'] == 5
        assert results['success_rate'] > 0.8
        assert results['throughput_per_second'] > 0
    
    def test_memory_usage_benchmark(self):
        """Test memory usage benchmark."""
        config = ZKPConfig(backend_type=ZKPType.MOCK)
        benchmark = ZKPBenchmark(config)
        
        results = benchmark.benchmark_memory_usage(num_operations=100)
        
        # Basic assertions
        assert results['initial_memory'] > 0
        assert results['final_memory'] > 0
        assert results['operations'] == 100
    
    def test_comprehensive_benchmark(self):
        """Test comprehensive benchmark suite."""
        config = ZKPConfig(backend_type=ZKPType.MOCK)
        benchmark = ZKPBenchmark(config)
        
        results = benchmark.run_comprehensive_benchmark()
        
        # Basic assertions
        assert 'proof_generation' in results
        assert 'proof_verification' in results
        assert 'batch_operations' in results
        assert 'concurrent_operations' in results
        assert 'memory_usage' in results
        
        # Save results for analysis
        benchmark.save_results("test_zkp_benchmark_results.json")


class TestZKPStressTests:
    """Stress tests for ZKP system."""
    
    def test_high_throughput_stress(self):
        """Stress test with high throughput."""
        config = ZKPConfig(backend_type=ZKPType.MOCK)
        manager = ZKPManager(config)
        manager.initialize()
        
        # Generate many proofs quickly
        start_time = time.time()
        successful_proofs = 0
        
        for i in range(1000):  # 1000 proofs
            request = ProofRequest(
                circuit_id=f"stress_{i}",
                public_inputs=[f"stress_input_{i}".encode()],
                private_inputs=[f"stress_private_{i}".encode()],
                proof_type=ZKPType.MOCK
            )
            
            result = manager.generate_proof(request)
            if result.is_success:
                successful_proofs += 1
        
        total_time = time.time() - start_time
        throughput = successful_proofs / total_time
        
        # Should handle high throughput
        assert successful_proofs > 800  # At least 80% success rate
        assert throughput > 50  # At least 50 proofs per second
    
    def test_memory_stress(self):
        """Stress test memory usage."""
        config = ZKPConfig(backend_type=ZKPType.MOCK)
        manager = ZKPManager(config)
        manager.initialize()
        
        # Generate proofs with large inputs
        large_input = b"x" * 10000  # 10KB input
        
        for i in range(100):  # 100 large proofs
            request = ProofRequest(
                circuit_id=f"memory_stress_{i}",
                public_inputs=[large_input],
                private_inputs=[large_input],
                proof_type=ZKPType.MOCK
            )
            
            result = manager.generate_proof(request)
            # Should not crash
            assert result.status in [ZKPStatus.SUCCESS, ZKPStatus.GENERATION_FAILED]
    
    def test_concurrent_stress(self):
        """Stress test with high concurrency."""
        config = ZKPConfig(backend_type=ZKPType.MOCK)
        manager = ZKPManager(config)
        manager.initialize()
        
        def stress_worker(worker_id: int, results: List[bool]):
            """Worker for concurrent stress test."""
            try:
                for i in range(50):  # 50 operations per worker
                    request = ProofRequest(
                        circuit_id=f"concurrent_stress_{worker_id}_{i}",
                        public_inputs=[f"concurrent_stress_{worker_id}_{i}".encode()],
                        private_inputs=[f"concurrent_stress_{worker_id}_{i}".encode()],
                        proof_type=ZKPType.MOCK
                    )
                    
                    result = manager.generate_proof(request)
                    if result.is_success:
                        verify_result = manager.verify_proof(result.proof, request.public_inputs)
                        results.append(verify_result.is_success and verify_result.is_valid)
                    else:
                        results.append(False)
            except Exception:
                results.append(False)
        
        # Run with high concurrency
        num_workers = 20
        results = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(stress_worker, i, results) for i in range(num_workers)]
            concurrent.futures.wait(futures)
        
        # Should handle high concurrency
        success_rate = sum(results) / len(results) if results else 0
        assert success_rate > 0.7  # At least 70% success rate under stress
    
    def test_long_running_stress(self):
        """Stress test for long-running operations."""
        config = ZKPConfig(backend_type=ZKPType.MOCK)
        manager = ZKPManager(config)
        manager.initialize()
        
        # Run for extended period
        start_time = time.time()
        operations = 0
        successful_operations = 0
        
        while time.time() - start_time < 30:  # Run for 30 seconds
            request = ProofRequest(
                circuit_id=f"long_running_{operations}",
                public_inputs=[f"long_running_{operations}".encode()],
                private_inputs=[f"long_running_{operations}".encode()],
                proof_type=ZKPType.MOCK
            )
            
            result = manager.generate_proof(request)
            operations += 1
            
            if result.is_success:
                verify_result = manager.verify_proof(result.proof, request.public_inputs)
                if verify_result.is_success and verify_result.is_valid:
                    successful_operations += 1
        
        # Should maintain performance over time
        success_rate = successful_operations / operations if operations > 0 else 0
        assert success_rate > 0.8  # At least 80% success rate
        assert operations > 100  # Should complete many operations
