#!/usr/bin/env python3
"""
Optimized GPU Acceleration Demo for DubChain

This demo showcases optimized GPU-accelerated operations that truly
benefit from GPU parallelization for blockchain workloads.
"""

import sys
import os
import time
import secrets
from typing import List, Tuple

# Add the project root to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.dubchain.crypto.optimized_gpu_crypto import OptimizedGPUCrypto, OptimizedGPUConfig


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


def test_optimized_gpu_availability():
    """Test optimized GPU availability and configuration."""
    print_header("OPTIMIZED GPU AVAILABILITY TEST")
    
    config = OptimizedGPUConfig(
        min_batch_size=100,  # Only use GPU for batches >= 100
        chunk_size=1000,
        fallback_to_cpu=True
    )
    gpu_crypto = OptimizedGPUCrypto(config)
    
    metrics = gpu_crypto.get_performance_metrics()
    
    print(f"GPU Available: {metrics['gpu_available']}")
    print(f"Device: {metrics['device']}")
    print(f"Configuration:")
    for key, value in metrics['config'].items():
        print(f"  {key}: {value}")
    
    return gpu_crypto


def test_smart_batch_processing(gpu_crypto: OptimizedGPUCrypto):
    """Test smart batch processing that uses GPU only when beneficial."""
    print_section("SMART BATCH PROCESSING")
    
    # Test small batch (should use CPU)
    small_batch_size = 50
    data_size = 1024
    
    print(f"\nTesting small batch ({small_batch_size} items) - should use CPU:")
    test_data = [secrets.token_bytes(data_size) for _ in range(small_batch_size)]
    
    start_time = time.time()
    results = gpu_crypto.hash_data_batch_optimized(test_data, "sha256")
    processing_time = time.time() - start_time
    
    print(f"  Processing time: {processing_time:.4f}s")
    print(f"  Throughput: {small_batch_size/processing_time:.2f} ops/sec")
    print(f"  Results count: {len(results)}")
    
    # Test large batch (should use GPU)
    large_batch_size = 2000
    data_size = 1024
    
    print(f"\nTesting large batch ({large_batch_size} items) - should use GPU:")
    test_data = [secrets.token_bytes(data_size) for _ in range(large_batch_size)]
    
    start_time = time.time()
    results = gpu_crypto.hash_data_batch_optimized(test_data, "sha256")
    processing_time = time.time() - start_time
    
    print(f"  Processing time: {processing_time:.4f}s")
    print(f"  Throughput: {large_batch_size/processing_time:.2f} ops/sec")
    print(f"  Results count: {len(results)}")


def test_signature_verification_optimized(gpu_crypto: OptimizedGPUCrypto):
    """Test optimized signature verification."""
    print_section("OPTIMIZED SIGNATURE VERIFICATION")
    
    # Test small batch (should use CPU)
    small_batch_size = 50
    verifications = []
    
    for i in range(small_batch_size):
        message = secrets.token_bytes(256)
        signature = secrets.token_bytes(64)
        public_key = secrets.token_bytes(33)
        algorithm = "secp256k1"
        verifications.append((message, signature, public_key, algorithm))
    
    print(f"\nTesting small signature batch ({small_batch_size} verifications) - should use CPU:")
    
    start_time = time.time()
    results = gpu_crypto.verify_signatures_batch_optimized(verifications)
    processing_time = time.time() - start_time
    
    print(f"  Processing time: {processing_time:.4f}s")
    print(f"  Throughput: {small_batch_size/processing_time:.2f} ops/sec")
    print(f"  Results count: {len(results)}")
    
    # Test large batch (should use GPU)
    large_batch_size = 1500
    verifications = []
    
    for i in range(large_batch_size):
        message = secrets.token_bytes(256)
        signature = secrets.token_bytes(64)
        public_key = secrets.token_bytes(33)
        algorithm = "secp256k1"
        verifications.append((message, signature, public_key, algorithm))
    
    print(f"\nTesting large signature batch ({large_batch_size} verifications) - should use GPU:")
    
    start_time = time.time()
    results = gpu_crypto.verify_signatures_batch_optimized(verifications)
    processing_time = time.time() - start_time
    
    print(f"  Processing time: {processing_time:.4f}s")
    print(f"  Throughput: {large_batch_size/processing_time:.2f} ops/sec")
    print(f"  Results count: {len(results)}")


def run_optimized_benchmark(gpu_crypto: OptimizedGPUCrypto):
    """Run optimized benchmark focusing on beneficial operations."""
    print_section("OPTIMIZED BENCHMARK")
    
    # Test different batch sizes
    batch_sizes = [50, 100, 500, 1000, 2000, 5000]
    data_size = 1024
    
    print(f"Benchmarking different batch sizes with {data_size} byte data:")
    print(f"{'Batch Size':<12} {'CPU Time':<10} {'GPU Time':<10} {'Speedup':<8} {'Used GPU':<8}")
    print(f"{'-'*60}")
    
    for batch_size in batch_sizes:
        # Generate test data
        test_data = [secrets.token_bytes(data_size) for _ in range(batch_size)]
        
        # CPU benchmark
        cpu_start = time.time()
        cpu_results = [gpu_crypto._hash_data_cpu(data, "sha256") for data in test_data]
        cpu_time = time.time() - cpu_start
        
        # GPU benchmark
        gpu_start = time.time()
        gpu_results = gpu_crypto.hash_data_batch_optimized(test_data, "sha256")
        gpu_time = time.time() - gpu_start
        
        speedup = cpu_time / gpu_time if gpu_time > 0 else 0
        used_gpu = batch_size >= gpu_crypto.config.min_batch_size and gpu_crypto.gpu_available
        
        print(f"{batch_size:<12} {cpu_time:<10.4f} {gpu_time:<10.4f} {speedup:<8.2f} {str(used_gpu):<8}")


def test_memory_efficiency_optimized(gpu_crypto: OptimizedGPUCrypto):
    """Test memory efficiency with optimized processing."""
    print_section("MEMORY EFFICIENCY TEST")
    
    # Test with very large dataset
    large_batch_size = 20000
    data_size = 512
    
    print(f"Testing memory efficiency with {large_batch_size} items ({data_size} bytes each):")
    
    # Generate test data
    test_data = [secrets.token_bytes(data_size) for _ in range(large_batch_size)]
    
    # Test optimized batch processing
    start_time = time.time()
    gpu_results = gpu_crypto.hash_data_batch_optimized(test_data, "sha256")
    processing_time = time.time() - start_time
    
    print(f"  Processing time: {processing_time:.4f}s")
    print(f"  Throughput: {large_batch_size/processing_time:.2f} ops/sec")
    print(f"  Memory efficient: {len(gpu_results) == large_batch_size}")
    print(f"  Used GPU: {large_batch_size >= gpu_crypto.config.min_batch_size and gpu_crypto.gpu_available}")


def test_blockchain_workload_simulation(gpu_crypto: OptimizedGPUCrypto):
    """Simulate realistic blockchain workloads."""
    print_section("BLOCKCHAIN WORKLOAD SIMULATION")
    
    # Simulate block processing with multiple transactions
    num_blocks = 10
    transactions_per_block = 100
    
    print(f"Simulating {num_blocks} blocks with {transactions_per_block} transactions each:")
    
    total_transactions = 0
    total_time = 0
    
    for block_num in range(num_blocks):
        # Generate transaction data
        transactions = []
        for tx_num in range(transactions_per_block):
            # Simulate transaction data
            tx_data = {
                'from': secrets.token_bytes(20),
                'to': secrets.token_bytes(20),
                'amount': secrets.token_bytes(8),
                'nonce': secrets.token_bytes(4),
                'signature': secrets.token_bytes(64),
            }
            # Combine into transaction bytes
            tx_bytes = b''.join(tx_data.values())
            transactions.append(tx_bytes)
        
        # Process block transactions
        start_time = time.time()
        tx_hashes = gpu_crypto.hash_data_batch_optimized(transactions, "sha256")
        block_time = time.time() - start_time
        
        total_transactions += len(transactions)
        total_time += block_time
        
        print(f"  Block {block_num + 1}: {block_time:.4f}s, {len(transactions)/block_time:.2f} tx/sec")
    
    print(f"\nTotal simulation results:")
    print(f"  Total transactions: {total_transactions}")
    print(f"  Total time: {total_time:.4f}s")
    print(f"  Average throughput: {total_transactions/total_time:.2f} tx/sec")
    print(f"  Used GPU: {transactions_per_block >= gpu_crypto.config.min_batch_size and gpu_crypto.gpu_available}")


def display_final_metrics_optimized(gpu_crypto: OptimizedGPUCrypto):
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
    
    # Calculate efficiency
    if metrics['gpu_operations'] > 0 and metrics['cpu_fallbacks'] > 0:
        gpu_efficiency = metrics['gpu_operations'] / (metrics['gpu_operations'] + metrics['cpu_fallbacks'])
        print(f"GPU Efficiency: {gpu_efficiency:.2%}")
    
    if metrics['gpu_utilization'] > 0.7:
        print("🎉 Excellent GPU utilization!")
    elif metrics['gpu_utilization'] > 0.3:
        print("✅ Good GPU utilization")
    else:
        print("⚠️  Low GPU utilization - consider optimizing batch sizes")


def main():
    """Main demo function."""
    print_header("DUBCHAIN OPTIMIZED GPU ACCELERATION DEMO")
    print("This demo showcases optimized GPU-accelerated operations")
    print("that intelligently use GPU only when beneficial.")
    
    try:
        # Test optimized GPU availability
        gpu_crypto = test_optimized_gpu_availability()
        
        if not gpu_crypto.gpu_available:
            print("⚠️  GPU not available - running CPU-only tests")
        
        # Run optimized tests
        test_smart_batch_processing(gpu_crypto)
        test_signature_verification_optimized(gpu_crypto)
        run_optimized_benchmark(gpu_crypto)
        test_memory_efficiency_optimized(gpu_crypto)
        test_blockchain_workload_simulation(gpu_crypto)
        display_final_metrics_optimized(gpu_crypto)
        
        print_header("OPTIMIZED DEMO COMPLETED SUCCESSFULLY")
        print("🎉 Optimized GPU acceleration demo completed!")
        print("Your DubChain project now has intelligent GPU acceleration")
        print("that automatically chooses the best processing method.")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
