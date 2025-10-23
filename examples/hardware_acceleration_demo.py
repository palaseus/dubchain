"""
Hardware acceleration demonstration for DubChain.

This example demonstrates the comprehensive hardware acceleration system
including CUDA, OpenCL, Metal, and CPU SIMD optimizations.
"""

import logging

logger = logging.getLogger(__name__)
import time
from typing import List, Dict, Any

from dubchain.hardware import (
    HardwareManager,
    HardwareManagerConfig,
    HardwareDetector,
)


def demo_hardware_detection():
    """Demonstrate hardware detection capabilities."""
    logger.info("🔍 Hardware Detection Demo")
    logger.info("=" * 50)
    
    # Detect hardware capabilities
    detector = HardwareDetector()
    capabilities = detector.detect_all()
    
    logger.info(f"Platform: {capabilities.platform.value}")
    logger.info(f"CPU: {capabilities.cpu.vendor} {capabilities.cpu.model}")
    logger.info(f"CPU Cores: {capabilities.cpu.cores} (Threads: {capabilities.cpu.threads})")
    logger.info(f"Memory: {capabilities.memory_gb:.1f} GB")
    logger.info(f"Available Accelerations: {[acc.value for acc in capabilities.available_accelerations]}")
    logger.info(f"Recommended: {capabilities.recommended_acceleration.value if capabilities.recommended_acceleration else 'None'}")
    
    if capabilities.gpus:
        logger.info(f"GPUs Found: {len(capabilities.gpus)}")
        for i, gpu in enumerate(capabilities.gpus):
            logger.info(f"  GPU {i}: {gpu.vendor} {gpu.name} ({gpu.memory_mb} MB)")
            logger.info(f"    Acceleration: {[acc.value for acc in gpu.acceleration_types]}")
    
    print()


def demo_hardware_acceleration():
    """Demonstrate hardware acceleration performance."""
    logger.info("🚀 Hardware Acceleration Demo")
    logger.info("=" * 50)
    
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
        
        logger.info(f"Testing with {len(test_data)} data items")
        print()
        
        # Benchmark all accelerators
        logger.info("Running benchmarks...")
        benchmark_results = manager.benchmark_accelerators(test_data, iterations=5)
        
        logger.info("Benchmark Results:")
        for acc_type, results in benchmark_results.items():
            if results.get("success", False):
                logger.info(f"  {acc_type}:")
                logger.info(f"    Avg Time: {results['avg_time_ms']:.2f} ms")
                logger.info(f"    Throughput: {results['throughput_ops_per_sec']:.0f} ops/sec")
                logger.info(f"    Min Time: {results['min_time_ms']:.2f} ms")
                logger.info(f"    Max Time: {results['max_time_ms']:.2f} ms")
            else:
                logger.info(f"  {acc_type}: Failed - {results.get('error', 'Unknown error')}")
        
        print()
        
        # Optimize configuration
        logger.info("Optimizing configuration...")
        optimization_results = manager.optimize_configuration()
        
        logger.info(f"Best Accelerator: {optimization_results.get('best_accelerator', 'None')}")
        logger.info(f"Best Throughput: {optimization_results.get('best_throughput', 0):.0f} ops/sec")
        logger.info(f"Current Accelerator: {optimization_results.get('current_accelerator', 'None')}")
        
        print()
        
        # Get performance stats
        stats = manager.get_performance_stats()
        logger.info("Performance Statistics:")
        logger.info(f"  Total Operations: {stats['manager']['total_operations']}")
        logger.info(f"  Total Time: {stats['manager']['total_time_ms']:.2f} ms")
        logger.info(f"  Average Time: {stats['manager']['avg_time_ms']:.2f} ms")
        logger.info(f"  Fallback Count: {stats['manager']['fallback_count']}")
        logger.info(f"  Available Accelerators: {stats['manager']['available_accelerators']}")
        
        print()


def demo_ethereum_integration():
    """Demonstrate Ethereum integration."""
    logger.info("⛓️ Ethereum Integration Demo")
    logger.info("=" * 50)
    
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
            logger.info("✅ Connected to Ethereum network")
            
            # Get network info
            network_info = client.get_network_info()
            logger.info(f"Chain ID: {network_info.get('chain_id', 'Unknown')}")
            logger.info(f"Latest Block: {network_info.get('latest_block', 0)}")
            logger.info(f"Gas Price: {network_info.get('gas_price_gwei', 0):.2f} Gwei")
            
            # Get gas prices
            gas_prices = network_info.get('gas_prices', {})
            logger.info("Gas Prices:")
            for strategy, price in gas_prices.items():
                logger.info(f"  {strategy}: {price:.2f} Gwei")
            
            # Get latest block
            latest_block = client.get_latest_block()
            if latest_block:
                logger.info(f"Latest Block Hash: {latest_block.hash}")
                logger.info(f"Block Timestamp: {latest_block.timestamp}")
                logger.info(f"Gas Used: {latest_block.gas_used:,}")
                logger.info(f"Gas Limit: {latest_block.gas_limit:,}")
                logger.info(f"Transactions: {len(latest_block.transactions)}")
            
        else:
            logger.info("❌ Failed to connect to Ethereum network")
            logger.info("   This is expected in demo mode without valid RPC endpoint")
        
    except ImportError as e:
        logger.info(f"❌ Ethereum integration not available: {e}")
        logger.info("   Install web3 package to enable Ethereum integration")
    
    print()


def demo_performance_profiling():
    """Demonstrate advanced performance profiling."""
    logger.info("📊 Performance Profiling Demo")
    logger.info("=" * 50)
    
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
        logger.info("Simulating operations...")
        
        for i in range(10):
            # Start operation
            op_context = profiler.start_operation(f"operation_{i}", {"iteration": i})
            
            # Simulate work
            time.sleep(0.1)
            
            # End operation
            profiler.end_operation(op_context, "success")
        
        # Get comprehensive stats
        stats = profiler.get_comprehensive_stats()
        
        logger.info("Performance Statistics:")
        logger.info(f"Memory Usage: {stats['memory'].get('current_rss_mb', 0):.1f} MB")
        logger.info(f"CPU Usage: {stats['cpu'].get('current_cpu_percent', 0):.1f}%")
        
        # Check for memory leaks
        leak_info = stats['memory_leak']
        if leak_info['leak_detected']:
            logger.info(f"⚠️ Potential memory leak detected!")
            logger.info(f"   Growth rate: {leak_info['growth_rate']:.2%}")
        else:
            logger.info("✅ No memory leaks detected")
        
        # Check for performance regressions
        regression_info = profiler.detect_performance_regression("operation_0")
        if regression_info['regression_detected']:
            logger.info(f"⚠️ Performance regression detected!")
            logger.info(f"   Regression factor: {regression_info['regression_factor']:.2f}")
        else:
            logger.info("✅ No performance regressions detected")
        
        # Generate report
        report_path = "performance_report.json"
        if profiler.generate_report(report_path):
            logger.info(f"📄 Performance report generated: {report_path}")
        
        profiler.cleanup()
        
    except ImportError as e:
        logger.info(f"❌ Performance profiling not available: {e}")
        logger.info("   Install opentelemetry and prometheus packages to enable profiling")
    
    print()


def main():
    """Main demo function."""
    logger.info("🎯 DubChain Hardware Acceleration & Ethereum Integration Demo")
    logger.info("=" * 70)
    print()
    
    # Run all demos
    demo_hardware_detection()
    demo_hardware_acceleration()
    demo_ethereum_integration()
    demo_performance_profiling()
    
    logger.info("🎉 Demo completed!")
    print()
    logger.info("Key Features Demonstrated:")
    logger.info("✅ Multi-platform hardware detection (CUDA, OpenCL, Metal, CPU SIMD)")
    logger.info("✅ Automatic hardware acceleration with fallback")
    logger.info("✅ Performance benchmarking and optimization")
    logger.info("✅ Ethereum Web3 integration with gas optimization")
    logger.info("✅ Advanced performance profiling and monitoring")
    logger.info("✅ Memory leak detection and regression analysis")
    print()
    logger.info("Next Steps:")
    logger.info("- Install additional dependencies for full functionality")
    logger.info("- Configure real blockchain RPC endpoints")
    logger.info("- Set up monitoring and alerting systems")
    logger.info("- Implement comprehensive testing suite")


if __name__ == "__main__":
    main()
