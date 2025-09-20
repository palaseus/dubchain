#!/usr/bin/env python3
"""
GPU Acceleration Demo for DubChain

This demo showcases GPU-accelerated cryptographic operations including:
- CUDA-accelerated hash computation
- Parallel signature verification on GPU
- Batch cryptographic operations
- Performance benchmarking
"""

import sys
import os
import time
import secrets
from typing import List, Tuple

# Add the project root to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.dubchain.crypto.gpu_crypto import GPUCrypto, GPUConfig


def print_header(title: str):
    """Print a formatted header."""
    print(f"\n{'='*60}")
    print(f"🚀 {title}")
    print(f"{'='*60}")


def print_section(title: str):
    """Print a formatted section header."""
    print(f"\n{'-'*40}")
    print(f"📊 {title}")
    print(f"{'-'*40}")


def test_gpu_availability():
    """Test GPU availability and configuration."""
    print_header("GPU AVAILABILITY TEST")
    
    config = GPUConfig()
    gpu_crypto = GPUCrypto(config)
    
    metrics = gpu_crypto.get_performance_metrics()
    
    print(f"GPU Available: {metrics['gpu_available']}")
    print(f"Device: {metrics['device']}")
    print(f"Configuration:")
    for key, value in metrics['config'].items():
        print(f"  {key}: {value}")
    
    return gpu_crypto


def test_single_hash_operations(gpu_crypto: GPUCrypto):
    """Test single hash operations."""
    print_section("SINGLE HASH OPERATIONS")
    
    # Test data of different sizes
    test_sizes = [64, 256, 1024, 4096, 16384]
    
    for size in test_sizes:
        print(f"\nTesting {size} byte data:")
        
        # Generate test data
        test_data = secrets.token_bytes(size)
        
        # Test GPU hash
        start_time = time.time()
        gpu_hash = gpu_crypto.hash_data_gpu(test_data, "sha256")
        gpu_time = time.time() - start_time
        
        # Test CPU hash
        start_time = time.time()
        cpu_hash = gpu_crypto._hash_data_cpu(test_data, "sha256")
        cpu_time = time.time() - start_time
        
        speedup = cpu_time / gpu_time if gpu_time > 0 else 0
        
        print(f"  GPU time: {gpu_time:.6f}s")
        print(f"  CPU time: {cpu_time:.6f}s")
        print(f"  Speedup: {speedup:.2f}x")
        print(f"  GPU hash: {gpu_hash[:16].hex()}...")
        print(f"  CPU hash: {cpu_hash[:16].hex()}...")


def test_batch_hash_operations(gpu_crypto: GPUCrypto):
    """Test batch hash operations."""
    print_section("BATCH HASH OPERATIONS")
    
    # Test different batch sizes
    batch_sizes = [10, 50, 100, 500, 1000]
    data_size = 1024
    
    for batch_size in batch_sizes:
        print(f"\nTesting batch of {batch_size} items ({data_size} bytes each):")
        
        # Generate test data
        test_data = [secrets.token_bytes(data_size) for _ in range(batch_size)]
        
        # Test GPU batch hash
        start_time = time.time()
        gpu_hashes = gpu_crypto.hash_data_batch_gpu(test_data, "sha256")
        gpu_time = time.time() - start_time
        
        # Test CPU batch hash
        start_time = time.time()
        cpu_hashes = [gpu_crypto._hash_data_cpu(data, "sha256") for data in test_data]
        cpu_time = time.time() - start_time
        
        speedup = cpu_time / gpu_time if gpu_time > 0 else 0
        
        print(f"  GPU time: {gpu_time:.4f}s")
        print(f"  CPU time: {cpu_time:.4f}s")
        print(f"  Speedup: {speedup:.2f}x")
        print(f"  GPU throughput: {batch_size/gpu_time:.2f} ops/sec")
        print(f"  CPU throughput: {batch_size/cpu_time:.2f} ops/sec")


def test_signature_verification(gpu_crypto: GPUCrypto):
    """Test signature verification operations."""
    print_section("SIGNATURE VERIFICATION")
    
    # Generate test signatures
    num_signatures = 100
    verifications = []
    
    for i in range(num_signatures):
        message = secrets.token_bytes(256)
        signature = secrets.token_bytes(64)
        public_key = secrets.token_bytes(33)
        algorithm = "secp256k1"
        verifications.append((message, signature, public_key, algorithm))
    
    print(f"Testing {num_signatures} signature verifications:")
    
    # Test GPU verification
    start_time = time.time()
    gpu_results = gpu_crypto.verify_signatures_gpu(verifications)
    gpu_time = time.time() - start_time
    
    # Test CPU verification
    start_time = time.time()
    cpu_results = gpu_crypto._verify_signatures_cpu(verifications)
    cpu_time = time.time() - start_time
    
    speedup = cpu_time / gpu_time if gpu_time > 0 else 0
    
    print(f"  GPU time: {gpu_time:.4f}s")
    print(f"  CPU time: {cpu_time:.4f}s")
    print(f"  Speedup: {speedup:.2f}x")
    print(f"  GPU throughput: {num_signatures/gpu_time:.2f} ops/sec")
    print(f"  CPU throughput: {num_signatures/cpu_time:.2f} ops/sec")
    print(f"  Results match: {gpu_results == cpu_results}")


def run_comprehensive_benchmark(gpu_crypto: GPUCrypto):
    """Run comprehensive benchmark."""
    print_section("COMPREHENSIVE BENCHMARK")
    
    # Test different data sizes
    data_sizes = [256, 1024, 4096, 16384]
    num_operations = 1000
    
    results = []
    
    for data_size in data_sizes:
        print(f"\nBenchmarking {data_size} byte data:")
        result = gpu_crypto.benchmark(data_size, num_operations)
        results.append(result)
    
    # Summary
    print(f"\n📈 BENCHMARK SUMMARY")
    print(f"{'Data Size':<12} {'CPU Time':<10} {'GPU Time':<10} {'Speedup':<8} {'GPU Throughput':<15}")
    print(f"{'-'*65}")
    
    for result in results:
        print(f"{result['data_size']:<12} {result['cpu_time']:<10.4f} {result['gpu_time']:<10.4f} "
              f"{result['speedup']:<8.2f} {result['gpu_throughput']:<15.2f}")


def test_memory_efficiency(gpu_crypto: GPUCrypto):
    """Test memory efficiency with large datasets."""
    print_section("MEMORY EFFICIENCY TEST")
    
    # Test with large dataset
    large_batch_size = 10000
    data_size = 512
    
    print(f"Testing memory efficiency with {large_batch_size} items ({data_size} bytes each):")
    
    # Generate test data
    test_data = [secrets.token_bytes(data_size) for _ in range(large_batch_size)]
    
    # Test GPU batch processing
    start_time = time.time()
    gpu_hashes = gpu_crypto.hash_data_batch_gpu(test_data, "sha256")
    gpu_time = time.time() - start_time
    
    print(f"  GPU batch processing time: {gpu_time:.4f}s")
    print(f"  GPU throughput: {large_batch_size/gpu_time:.2f} ops/sec")
    print(f"  Memory efficient: {len(gpu_hashes) == large_batch_size}")


def display_final_metrics(gpu_crypto: GPUCrypto):
    """Display final performance metrics."""
    print_section("FINAL PERFORMANCE METRICS")
    
    metrics = gpu_crypto.get_performance_metrics()
    
    print(f"Total Operations: {metrics['total_operations']}")
    print(f"GPU Operations: {metrics['gpu_operations']}")
    print(f"CPU Fallbacks: {metrics['cpu_fallbacks']}")
    print(f"Batch Operations: {metrics['batch_operations']}")
    print(f"GPU Utilization: {metrics['gpu_utilization']:.2%}")
    print(f"Average GPU Time: {metrics['avg_gpu_time']:.6f}s")
    print(f"Average CPU Time: {metrics['avg_cpu_time']:.6f}s")
    
    if metrics['gpu_utilization'] > 0.5:
        print("🎉 Excellent GPU utilization!")
    elif metrics['gpu_utilization'] > 0.2:
        print("✅ Good GPU utilization")
    else:
        print("⚠️  Low GPU utilization - consider optimizing batch sizes")


def main():
    """Main demo function."""
    print_header("DUBCHAIN GPU ACCELERATION DEMO")
    print("This demo showcases GPU-accelerated cryptographic operations")
    print("using CUDA for enhanced blockchain performance.")
    
    try:
        # Test GPU availability
        gpu_crypto = test_gpu_availability()
        
        if not gpu_crypto.gpu_available:
            print("⚠️  GPU not available - running CPU-only tests")
        
        # Run tests
        test_single_hash_operations(gpu_crypto)
        test_batch_hash_operations(gpu_crypto)
        test_signature_verification(gpu_crypto)
        run_comprehensive_benchmark(gpu_crypto)
        test_memory_efficiency(gpu_crypto)
        display_final_metrics(gpu_crypto)
        
        print_header("DEMO COMPLETED SUCCESSFULLY")
        print("🎉 GPU acceleration demo completed!")
        print("Your DubChain project now has GPU-accelerated cryptographic operations.")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
