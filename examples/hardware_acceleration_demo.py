"""
Hardware acceleration demonstration for DubChain.

This example demonstrates the comprehensive hardware acceleration system
including CUDA, OpenCL, Metal, and CPU SIMD optimizations.
"""

import time
from typing import List, Dict, Any

from dubchain.hardware import (
    HardwareManager,
    HardwareManagerConfig,
    HardwareDetector,
)


def demo_hardware_detection():
    """Demonstrate hardware detection capabilities."""
    print("🔍 Hardware Detection Demo")
    print("=" * 50)
    
    # Detect hardware capabilities
    detector = HardwareDetector()
    capabilities = detector.detect_all()
    
    print(f"Platform: {capabilities.platform.value}")
    print(f"CPU: {capabilities.cpu.vendor} {capabilities.cpu.model}")
    print(f"CPU Cores: {capabilities.cpu.cores} (Threads: {capabilities.cpu.threads})")
    print(f"Memory: {capabilities.memory_gb:.1f} GB")
    print(f"Available Accelerations: {[acc.value for acc in capabilities.available_accelerations]}")
    print(f"Recommended: {capabilities.recommended_acceleration.value if capabilities.recommended_acceleration else 'None'}")
    
    if capabilities.gpus:
        print(f"GPUs Found: {len(capabilities.gpus)}")
        for i, gpu in enumerate(capabilities.gpus):
            print(f"  GPU {i}: {gpu.vendor} {gpu.name} ({gpu.memory_mb} MB)")
            print(f"    Acceleration: {[acc.value for acc in gpu.acceleration_types]}")
    
    print()


def demo_hardware_acceleration():
    """Demonstrate hardware acceleration performance."""
    print("🚀 Hardware Acceleration Demo")
    print("=" * 50)
    
    # Initialize hardware manager
    config = HardwareManagerConfig(
        enable_cuda=True,
        enable_opencl=True,
        enable_cpu_simd=True,
        fallback_enabled=True,
        performance_monitoring=True,
    )
    
    with HardwareManager(config) as manager:
        # Generate test data
        test_data = [f"test_data_{i}".encode() for i in range(1000)]
        
        print(f"Testing with {len(test_data)} data items")
        print()
        
        # Benchmark all accelerators
        print("Running benchmarks...")
        benchmark_results = manager.benchmark_accelerators(test_data, iterations=5)
        
        print("Benchmark Results:")
        for acc_type, results in benchmark_results.items():
            if results.get("success", False):
                print(f"  {acc_type}:")
                print(f"    Avg Time: {results['avg_time_ms']:.2f} ms")
                print(f"    Throughput: {results['throughput_ops_per_sec']:.0f} ops/sec")
                print(f"    Min Time: {results['min_time_ms']:.2f} ms")
                print(f"    Max Time: {results['max_time_ms']:.2f} ms")
            else:
                print(f"  {acc_type}: Failed - {results.get('error', 'Unknown error')}")
        
        print()
        
        # Optimize configuration
        print("Optimizing configuration...")
        optimization_results = manager.optimize_configuration()
        
        print(f"Best Accelerator: {optimization_results.get('best_accelerator', 'None')}")
        print(f"Best Throughput: {optimization_results.get('best_throughput', 0):.0f} ops/sec")
        print(f"Current Accelerator: {optimization_results.get('current_accelerator', 'None')}")
        
        print()
        
        # Get performance stats
        stats = manager.get_performance_stats()
        print("Performance Statistics:")
        print(f"  Total Operations: {stats['manager']['total_operations']}")
        print(f"  Total Time: {stats['manager']['total_time_ms']:.2f} ms")
        print(f"  Average Time: {stats['manager']['avg_time_ms']:.2f} ms")
        print(f"  Fallback Count: {stats['manager']['fallback_count']}")
        print(f"  Available Accelerators: {stats['manager']['available_accelerators']}")
        
        print()


def demo_ethereum_integration():
    """Demonstrate Ethereum integration."""
    print("⛓️ Ethereum Integration Demo")
    print("=" * 50)
    
    try:
        from dubchain.bridge.chains.ethereum import EthereumClient, EthereumConfig
        
        # Initialize Ethereum client
        config = EthereumConfig(
            rpc_url="https://eth-mainnet.g.alchemy.com/v2/demo",  # Demo URL
            chain_id=1,
            gas_price_strategy="standard",
            enable_gas_price_oracle=True,
        )
        
        client = EthereumClient(config)
        
        if client.is_connected():
            print("✅ Connected to Ethereum network")
            
            # Get network info
            network_info = client.get_network_info()
            print(f"Chain ID: {network_info.get('chain_id', 'Unknown')}")
            print(f"Latest Block: {network_info.get('latest_block', 0)}")
            print(f"Gas Price: {network_info.get('gas_price_gwei', 0):.2f} Gwei")
            
            # Get gas prices
            gas_prices = network_info.get('gas_prices', {})
            print("Gas Prices:")
            for strategy, price in gas_prices.items():
                print(f"  {strategy}: {price:.2f} Gwei")
            
            # Get latest block
            latest_block = client.get_latest_block()
            if latest_block:
                print(f"Latest Block Hash: {latest_block.hash}")
                print(f"Block Timestamp: {latest_block.timestamp}")
                print(f"Gas Used: {latest_block.gas_used:,}")
                print(f"Gas Limit: {latest_block.gas_limit:,}")
                print(f"Transactions: {len(latest_block.transactions)}")
            
        else:
            print("❌ Failed to connect to Ethereum network")
            print("   This is expected in demo mode without valid RPC endpoint")
        
    except ImportError as e:
        print(f"❌ Ethereum integration not available: {e}")
        print("   Install web3 package to enable Ethereum integration")
    
    print()


def demo_performance_profiling():
    """Demonstrate advanced performance profiling."""
    print("📊 Performance Profiling Demo")
    print("=" * 50)
    
    try:
        from dubchain.performance.tracing import (
            AdvancedProfiler,
            TraceConfig,
            ProfilingConfig,
            MetricsConfig,
        )
        
        # Initialize profiler
        trace_config = TraceConfig(
            enable_tracing=True,
            service_name="dubchain_demo",
        )
        
        profiling_config = ProfilingConfig(
            enable_profiling=True,
            enable_memory_profiling=True,
            enable_cpu_profiling=True,
            profiling_interval=0.5,
        )
        
        metrics_config = MetricsConfig(
            enable_prometheus=False,  # Disable for demo
            enable_custom_metrics=True,
        )
        
        profiler = AdvancedProfiler(trace_config, profiling_config, metrics_config)
        
        # Simulate some operations
        print("Simulating operations...")
        
        for i in range(10):
            # Start operation
            op_context = profiler.start_operation(f"operation_{i}", {"iteration": i})
            
            # Simulate work
            time.sleep(0.1)
            
            # End operation
            profiler.end_operation(op_context, "success")
        
        # Get comprehensive stats
        stats = profiler.get_comprehensive_stats()
        
        print("Performance Statistics:")
        print(f"Memory Usage: {stats['memory'].get('current_rss_mb', 0):.1f} MB")
        print(f"CPU Usage: {stats['cpu'].get('current_cpu_percent', 0):.1f}%")
        
        # Check for memory leaks
        leak_info = stats['memory_leak']
        if leak_info['leak_detected']:
            print(f"⚠️ Potential memory leak detected!")
            print(f"   Growth rate: {leak_info['growth_rate']:.2%}")
        else:
            print("✅ No memory leaks detected")
        
        # Check for performance regressions
        regression_info = profiler.detect_performance_regression("operation_0")
        if regression_info['regression_detected']:
            print(f"⚠️ Performance regression detected!")
            print(f"   Regression factor: {regression_info['regression_factor']:.2f}")
        else:
            print("✅ No performance regressions detected")
        
        # Generate report
        report_path = "performance_report.json"
        if profiler.generate_report(report_path):
            print(f"📄 Performance report generated: {report_path}")
        
        profiler.cleanup()
        
    except ImportError as e:
        print(f"❌ Performance profiling not available: {e}")
        print("   Install opentelemetry and prometheus packages to enable profiling")
    
    print()


def main():
    """Main demo function."""
    print("🎯 DubChain Hardware Acceleration & Ethereum Integration Demo")
    print("=" * 70)
    print()
    
    # Run all demos
    demo_hardware_detection()
    demo_hardware_acceleration()
    demo_ethereum_integration()
    demo_performance_profiling()
    
    print("🎉 Demo completed!")
    print()
    print("Key Features Demonstrated:")
    print("✅ Multi-platform hardware detection (CUDA, OpenCL, Metal, CPU SIMD)")
    print("✅ Automatic hardware acceleration with fallback")
    print("✅ Performance benchmarking and optimization")
    print("✅ Ethereum Web3 integration with gas optimization")
    print("✅ Advanced performance profiling and monitoring")
    print("✅ Memory leak detection and regression analysis")
    print()
    print("Next Steps:")
    print("- Install additional dependencies for full functionality")
    print("- Configure real blockchain RPC endpoints")
    print("- Set up monitoring and alerting systems")
    print("- Implement comprehensive testing suite")


if __name__ == "__main__":
    main()
