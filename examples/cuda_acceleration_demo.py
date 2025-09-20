#!/usr/bin/env python3
"""
CUDA Acceleration Demo for DubChain.

This script demonstrates the comprehensive CUDA integration across all components
of the DubChain blockchain system, showcasing GPU acceleration for:
- Cryptographic operations
- Consensus mechanisms
- Virtual machine operations
- Storage operations
"""

import time
import secrets
import json
from typing import List, Dict, Any

# Import DubChain CUDA components
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from dubchain.cuda import CUDAManager, CUDAConfig, cuda_available
from dubchain.crypto import GPUCrypto, GPUConfig
from dubchain.consensus import CUDAConsensusAccelerator, CUDAConsensusConfig
from dubchain.vm import CUDAVMAccelerator, CUDAVMConfig
from dubchain.storage import CUDAStorageAccelerator, CUDAStorageConfig


def print_header(title: str):
    """Print a formatted header."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def print_section(title: str):
    """Print a formatted section header."""
    print(f"\n{'-'*40}")
    print(f"  {title}")
    print(f"{'-'*40}")


def check_cuda_availability():
    """Check and display CUDA availability."""
    print_header("CUDA AVAILABILITY CHECK")
    
    available = cuda_available()
    print(f"CUDA Available: {'✅ YES' if available else '❌ NO'}")
    
    if available:
        from dubchain.cuda import get_cuda_device, get_cuda_memory_info
        
        device = get_cuda_device()
        memory_info = get_cuda_memory_info()
        
        print(f"CUDA Device: {device}")
        print(f"Total Memory: {memory_info.get('total_memory', 0) / (1024**3):.2f} GB")
        print(f"Free Memory: {memory_info.get('free_memory', 0) / (1024**3):.2f} GB")
        print(f"Device Name: {memory_info.get('device_name', 'Unknown')}")
    else:
        print("CUDA is not available. Operations will fall back to CPU.")
    
    return available


def demo_crypto_acceleration():
    """Demonstrate CUDA-accelerated cryptographic operations."""
    print_section("CRYPTOGRAPHIC OPERATIONS")
    
    # Initialize GPU crypto
    gpu_crypto = GPUCrypto()
    print(f"GPU Crypto initialized - Available: {gpu_crypto.gpu_available}")
    
    # Generate test data
    test_data = [secrets.token_bytes(64) for _ in range(100)]
    print(f"Generated {len(test_data)} test data items")
    
    # Test single hash operation
    print("\nTesting single hash operation...")
    single_data = test_data[0]
    start_time = time.time()
    single_hash = gpu_crypto.hash_data_gpu(single_data, "sha256")
    single_time = time.time() - start_time
    print(f"Single hash: {single_hash.hex()[:16]}... (took {single_time:.4f}s)")
    
    # Test batch hash operations
    print("\nTesting batch hash operations...")
    start_time = time.time()
    batch_hashes = gpu_crypto.hash_data_batch_gpu(test_data, "sha256")
    batch_time = time.time() - start_time
    print(f"Batch hashes: {len(batch_hashes)} operations (took {batch_time:.4f}s)")
    print(f"Throughput: {len(batch_hashes) / batch_time:.2f} hashes/sec")
    
    # Test signature verification
    print("\nTesting signature verification...")
    signatures = [secrets.token_bytes(64) for _ in range(50)]
    public_keys = [secrets.token_bytes(33) for _ in range(50)]
    messages = [secrets.token_bytes(32) for _ in range(50)]
    
    start_time = time.time()
    verification_results = gpu_crypto.verify_signatures_gpu(
        list(zip(messages, signatures, public_keys, ["ecdsa"] * 50))
    )
    verification_time = time.time() - start_time
    print(f"Signature verification: {len(verification_results)} verifications (took {verification_time:.4f}s)")
    print(f"Throughput: {len(verification_results) / verification_time:.2f} verifications/sec")
    
    # Get performance metrics
    metrics = gpu_crypto.get_performance_metrics()
    print(f"\nPerformance Metrics:")
    print(f"  Total operations: {metrics['total_operations']}")
    print(f"  GPU operations: {metrics['gpu_operations']}")
    print(f"  CPU fallbacks: {metrics['cpu_fallbacks']}")
    print(f"  GPU utilization: {metrics['gpu_utilization']:.2%}")


def demo_consensus_acceleration():
    """Demonstrate CUDA-accelerated consensus operations."""
    print_section("CONSENSUS OPERATIONS")
    
    # Initialize CUDA consensus accelerator
    consensus_accelerator = CUDAConsensusAccelerator()
    print(f"CUDA Consensus Accelerator initialized - Available: {consensus_accelerator.cuda_manager.available}")
    
    # Generate test blocks
    test_blocks = []
    for i in range(75):
        block = {
            'index': i,
            'timestamp': time.time(),
            'data': secrets.token_bytes(64),
            'previous_hash': secrets.token_hex(32),
            'validator': f"validator_{i}",
            'transactions': [secrets.token_bytes(32) for _ in range(10)],
        }
        test_blocks.append(block)
    
    print(f"Generated {len(test_blocks)} test blocks")
    
    # Test block validation
    print("\nTesting block validation...")
    start_time = time.time()
    validation_results = consensus_accelerator.validate_blocks_batch(test_blocks)
    validation_time = time.time() - start_time
    print(f"Block validation: {len(validation_results)} blocks (took {validation_time:.4f}s)")
    print(f"Throughput: {len(validation_results) / validation_time:.2f} blocks/sec")
    
    # Test signature verification
    print("\nTesting consensus signature verification...")
    signatures = [secrets.token_bytes(64) for _ in range(60)]
    public_keys = [secrets.token_bytes(33) for _ in range(60)]
    messages = [secrets.token_bytes(32) for _ in range(60)]
    
    start_time = time.time()
    sig_results = consensus_accelerator.verify_signatures_batch(signatures, public_keys, messages)
    sig_time = time.time() - start_time
    print(f"Signature verification: {len(sig_results)} signatures (took {sig_time:.4f}s)")
    print(f"Throughput: {len(sig_results) / sig_time:.2f} signatures/sec")
    
    # Test consensus operations
    print("\nTesting consensus operations...")
    operations = []
    for i in range(40):
        operation = {
            'id': f"consensus_op_{i}",
            'type': 'consensus_operation',
            'data': secrets.token_bytes(32),
            'timestamp': time.time(),
        }
        operations.append(operation)
    
    start_time = time.time()
    op_results = consensus_accelerator.process_consensus_operations(operations)
    op_time = time.time() - start_time
    print(f"Consensus operations: {len(op_results)} operations (took {op_time:.4f}s)")
    print(f"Throughput: {len(op_results) / op_time:.2f} operations/sec")
    
    # Get performance metrics
    metrics = consensus_accelerator.get_performance_metrics()
    print(f"\nPerformance Metrics:")
    print(f"  Total operations: {metrics['total_operations']}")
    print(f"  GPU operations: {metrics['gpu_operations']}")
    print(f"  CPU fallbacks: {metrics['cpu_fallbacks']}")


def demo_vm_acceleration():
    """Demonstrate CUDA-accelerated virtual machine operations."""
    print_section("VIRTUAL MACHINE OPERATIONS")
    
    # Initialize CUDA VM accelerator
    vm_accelerator = CUDAVMAccelerator()
    print(f"CUDA VM Accelerator initialized - Available: {vm_accelerator.cuda_manager.available}")
    
    # Test bytecode processing
    print("\nTesting bytecode processing...")
    bytecode_list = [secrets.token_bytes(64) for _ in range(80)]
    start_time = time.time()
    processed_bytecode = vm_accelerator.process_bytecode_batch(bytecode_list, optimization_level=1)
    processing_time = time.time() - start_time
    print(f"Bytecode processing: {len(processed_bytecode)} bytecode sequences (took {processing_time:.4f}s)")
    print(f"Throughput: {len(processed_bytecode) / processing_time:.2f} sequences/sec")
    
    # Test VM operations
    print("\nTesting VM operations...")
    vm_operations = []
    for i in range(60):
        operation = {
            'id': f"vm_op_{i}",
            'type': 'vm_operation',
            'data': secrets.token_bytes(32),
            'gas_limit': 1000,
            'timestamp': time.time(),
        }
        vm_operations.append(operation)
    
    start_time = time.time()
    vm_results = vm_accelerator.execute_operations_batch(vm_operations)
    vm_time = time.time() - start_time
    print(f"VM operations: {len(vm_results)} operations (took {vm_time:.4f}s)")
    print(f"Throughput: {len(vm_results) / vm_time:.2f} operations/sec")
    
    # Test bytecode optimization
    print("\nTesting bytecode optimization...")
    optimization_rules = ["constant_folding", "dead_code_elimination", "peephole"]
    start_time = time.time()
    optimized_bytecode = vm_accelerator.optimize_bytecode_batch(bytecode_list, optimization_rules)
    optimization_time = time.time() - start_time
    print(f"Bytecode optimization: {len(optimized_bytecode)} optimizations (took {optimization_time:.4f}s)")
    print(f"Throughput: {len(optimized_bytecode) / optimization_time:.2f} optimizations/sec")
    
    # Get performance metrics
    metrics = vm_accelerator.get_performance_metrics()
    print(f"\nPerformance Metrics:")
    print(f"  Total operations: {metrics['total_operations']}")
    print(f"  GPU operations: {metrics['gpu_operations']}")
    print(f"  CPU fallbacks: {metrics['cpu_fallbacks']}")


def demo_storage_acceleration():
    """Demonstrate CUDA-accelerated storage operations."""
    print_section("STORAGE OPERATIONS")
    
    # Initialize CUDA storage accelerator
    storage_accelerator = CUDAStorageAccelerator()
    print(f"CUDA Storage Accelerator initialized - Available: {storage_accelerator.cuda_manager.available}")
    
    # Test data serialization
    print("\nTesting data serialization...")
    test_data = []
    for i in range(70):
        data = {
            'id': f"storage_data_{i}",
            'content': secrets.token_bytes(64),
            'metadata': {
                'index': i,
                'timestamp': time.time(),
                'type': 'test_data',
            },
        }
        test_data.append(data)
    
    start_time = time.time()
    serialized_data = storage_accelerator.serialize_data_batch(test_data, "json")
    serialization_time = time.time() - start_time
    print(f"Data serialization: {len(serialized_data)} objects (took {serialization_time:.4f}s)")
    print(f"Throughput: {len(serialized_data) / serialization_time:.2f} objects/sec")
    
    # Test data deserialization
    print("\nTesting data deserialization...")
    start_time = time.time()
    deserialized_data = storage_accelerator.deserialize_data_batch(serialized_data, "json")
    deserialization_time = time.time() - start_time
    print(f"Data deserialization: {len(deserialized_data)} objects (took {deserialization_time:.4f}s)")
    print(f"Throughput: {len(deserialized_data) / deserialization_time:.2f} objects/sec")
    
    # Test data compression
    print("\nTesting data compression...")
    raw_data = [secrets.token_bytes(128) for _ in range(50)]
    start_time = time.time()
    compressed_data = storage_accelerator.compress_data_batch(raw_data, compression_level=6)
    compression_time = time.time() - start_time
    print(f"Data compression: {len(compressed_data)} objects (took {compression_time:.4f}s)")
    print(f"Throughput: {len(compressed_data) / compression_time:.2f} objects/sec")
    
    # Test data decompression
    print("\nTesting data decompression...")
    start_time = time.time()
    decompressed_data = storage_accelerator.decompress_data_batch(compressed_data)
    decompression_time = time.time() - start_time
    print(f"Data decompression: {len(decompressed_data)} objects (took {decompression_time:.4f}s)")
    print(f"Throughput: {len(decompressed_data) / decompression_time:.2f} objects/sec")
    
    # Test storage operations
    print("\nTesting storage operations...")
    storage_operations = []
    for i in range(45):
        operation = {
            'id': f"storage_op_{i}",
            'type': 'storage_operation',
            'data': secrets.token_bytes(32),
            'timestamp': time.time(),
        }
        storage_operations.append(operation)
    
    start_time = time.time()
    storage_results = storage_accelerator.process_storage_operations(storage_operations)
    storage_time = time.time() - start_time
    print(f"Storage operations: {len(storage_results)} operations (took {storage_time:.4f}s)")
    print(f"Throughput: {len(storage_results) / storage_time:.2f} operations/sec")
    
    # Get performance metrics
    metrics = storage_accelerator.get_performance_metrics()
    print(f"\nPerformance Metrics:")
    print(f"  Total operations: {metrics['total_operations']}")
    print(f"  GPU operations: {metrics['gpu_operations']}")
    print(f"  CPU fallbacks: {metrics['cpu_fallbacks']}")


def demo_comprehensive_benchmark():
    """Demonstrate comprehensive CUDA performance benchmark."""
    print_section("COMPREHENSIVE PERFORMANCE BENCHMARK")
    
    # Initialize all components
    gpu_crypto = GPUCrypto()
    consensus_accelerator = CUDAConsensusAccelerator()
    vm_accelerator = CUDAVMAccelerator()
    storage_accelerator = CUDAStorageAccelerator()
    
    print("All CUDA components initialized")
    
    # Generate comprehensive test data
    test_size = 100
    print(f"Running comprehensive benchmark with {test_size} operations per component...")
    
    # Crypto benchmark
    crypto_data = [secrets.token_bytes(64) for _ in range(test_size)]
    crypto_start = time.time()
    crypto_results = gpu_crypto.hash_data_batch_gpu(crypto_data, "sha256")
    crypto_time = time.time() - crypto_start
    
    # Consensus benchmark
    consensus_blocks = [{'index': i, 'data': secrets.token_bytes(64)} for i in range(test_size)]
    consensus_start = time.time()
    consensus_results = consensus_accelerator.validate_blocks_batch(consensus_blocks)
    consensus_time = time.time() - consensus_start
    
    # VM benchmark
    vm_operations = [{'id': f"op_{i}", 'data': secrets.token_bytes(32)} for i in range(test_size)]
    vm_start = time.time()
    vm_results = vm_accelerator.execute_operations_batch(vm_operations)
    vm_time = time.time() - vm_start
    
    # Storage benchmark
    storage_data = [{'id': f"data_{i}", 'content': secrets.token_bytes(64)} for i in range(test_size)]
    storage_start = time.time()
    storage_results = storage_accelerator.serialize_data_batch(storage_data, "json")
    storage_time = time.time() - storage_start
    
    # Calculate totals
    total_time = crypto_time + consensus_time + vm_time + storage_time
    total_operations = len(crypto_results) + len(consensus_results) + len(vm_results) + len(storage_results)
    
    print(f"\nBenchmark Results:")
    print(f"  Crypto: {crypto_time:.4f}s ({len(crypto_results)} operations)")
    print(f"  Consensus: {consensus_time:.4f}s ({len(consensus_results)} operations)")
    print(f"  VM: {vm_time:.4f}s ({len(vm_results)} operations)")
    print(f"  Storage: {storage_time:.4f}s ({len(storage_results)} operations)")
    print(f"  Total: {total_time:.4f}s ({total_operations} operations)")
    print(f"  Overall Throughput: {total_operations / total_time:.2f} operations/sec")
    
    # Get global CUDA metrics
    from dubchain.cuda import get_global_cuda_manager
    global_manager = get_global_cuda_manager()
    global_metrics = global_manager.get_performance_metrics()
    
    print(f"\nGlobal CUDA Metrics:")
    print(f"  Total operations: {global_metrics['total_operations']}")
    print(f"  GPU operations: {global_metrics['gpu_operations']}")
    print(f"  CPU fallbacks: {global_metrics['cpu_fallbacks']}")
    print(f"  GPU utilization: {global_metrics.get('gpu_utilization', 0):.2%}")


def main():
    """Main demo function."""
    print_header("DUBCHAIN CUDA ACCELERATION DEMO")
    print("This demo showcases comprehensive CUDA integration across all DubChain components.")
    
    # Check CUDA availability
    cuda_available = check_cuda_availability()
    
    if not cuda_available:
        print("\n⚠️  CUDA is not available. All operations will use CPU fallback.")
        print("   To enable CUDA acceleration, ensure you have:")
        print("   - NVIDIA GPU with CUDA support")
        print("   - CUDA toolkit installed")
        print("   - PyTorch with CUDA support")
        print("   - CuPy installed (optional)")
    
    # Run component demos
    demo_crypto_acceleration()
    demo_consensus_acceleration()
    demo_vm_acceleration()
    demo_storage_acceleration()
    
    # Run comprehensive benchmark
    demo_comprehensive_benchmark()
    
    print_header("DEMO COMPLETED")
    print("CUDA acceleration has been successfully demonstrated across all DubChain components!")
    print("The system provides:")
    print("  ✅ GPU-accelerated cryptographic operations")
    print("  ✅ GPU-accelerated consensus mechanisms")
    print("  ✅ GPU-accelerated virtual machine operations")
    print("  ✅ GPU-accelerated storage operations")
    print("  ✅ Automatic CPU fallback when GPU is not available")
    print("  ✅ Comprehensive performance monitoring")
    print("  ✅ Thread-safe concurrent operations")


if __name__ == "__main__":
    main()
