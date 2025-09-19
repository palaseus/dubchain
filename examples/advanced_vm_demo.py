#!/usr/bin/env python3
"""
Advanced Virtual Machine Demo for DubChain

This demo showcases the sophisticated VM enhancements including:
- Advanced opcodes and operations
- Just-in-time compilation
- Parallel execution support
- Advanced gas optimization
- Performance monitoring
- Execution caching

Run this demo to see how DubChain's VM provides enterprise-grade
performance and advanced execution capabilities.
"""

import asyncio
import time
import json
from typing import Dict, Any, List

# Import DubChain VM components
from dubchain.vm import (
    SmartContract,
    ContractType,
    ContractState,
    ContractStorage,
    ContractMemory,
    GasMeter,
    GasCost,
    OpcodeEnum
)
from dubchain.vm.advanced_execution_engine import AdvancedExecutionEngine, ExecutionMetrics
from dubchain.vm.advanced_opcodes import AdvancedOpcodeEnum, advanced_opcode_registry
from dubchain.crypto.hashing import SHA256Hasher


class AdvancedVMDemo:
    """Demonstrates advanced VM capabilities."""
    
    def __init__(self):
        """Initialize the demo."""
        self.advanced_engine = AdvancedExecutionEngine()
        self.demo_contracts = []
        self.performance_data = {}
    
    def setup_demo_contracts(self) -> None:
        """Setup demo smart contracts."""
        print("📝 Setting up demo smart contracts...")
        
        # Contract 1: Simple arithmetic operations
        arithmetic_contract = SmartContract(
            address="0x1111111111111111111111111111111111111111",
            bytecode=b"\x60\x01\x60\x02\x01\x60\x03\x02\x60\x04\x03",  # Simple arithmetic
            contract_type=ContractType.STANDARD,
            name="Arithmetic Contract",
            version="1.0.0"
        )
        self.demo_contracts.append(arithmetic_contract)
        print("  ✅ Arithmetic contract created")
        
        # Contract 2: Memory operations
        memory_contract = SmartContract(
            address="0x2222222222222222222222222222222222222222",
            bytecode=b"\x60\x20\x60\x00\x52\x60\x40\x60\x20\x52",  # Memory operations
            contract_type=ContractType.STANDARD,
            name="Memory Contract",
            version="1.0.0"
        )
        self.demo_contracts.append(memory_contract)
        print("  ✅ Memory contract created")
        
        # Contract 3: Storage operations
        storage_contract = SmartContract(
            address="0x3333333333333333333333333333333333333333",
            bytecode=b"\x60\x01\x60\x00\x55\x60\x00\x54",  # Storage operations
            contract_type=ContractType.STANDARD,
            name="Storage Contract",
            version="1.0.0"
        )
        self.demo_contracts.append(storage_contract)
        print("  ✅ Storage contract created")
        
        # Contract 4: Complex operations
        complex_contract = SmartContract(
            address="0x4444444444444444444444444444444444444444",
            bytecode=b"\x60\x01\x60\x02\x01\x60\x03\x02\x60\x04\x03\x60\x05\x04",  # Complex operations
            contract_type=ContractType.STANDARD,
            name="Complex Contract",
            version="1.0.0"
        )
        self.demo_contracts.append(complex_contract)
        print("  ✅ Complex contract created")
    
    def demonstrate_advanced_opcodes(self) -> None:
        """Demonstrate advanced opcodes."""
        print("\n🔧 ADVANCED OPCODES DEMONSTRATION")
        print("=" * 50)
        
        # Show available advanced opcodes
        print("📊 Available Advanced Opcodes:")
        
        opcode_categories = {
            'Cryptographic': [],
            'Memory': [],
            'Storage': [],
            'Parallel': [],
            'Gas Optimization': [],
            'Math': [],
            'String': [],
            'Array': [],
            'JSON': [],
            'Time': [],
            'Random': [],
            'Debug': []
        }
        
        for opcode, info in advanced_opcode_registry.get_all_opcodes().items():
            if 'CRYPTO' in info.name or 'HASH' in info.name:
                opcode_categories['Cryptographic'].append(info.name)
            elif 'MEM' in info.name:
                opcode_categories['Memory'].append(info.name)
            elif 'STORE' in info.name or 'LOAD' in info.name:
                opcode_categories['Storage'].append(info.name)
            elif 'PARALLEL' in info.name:
                opcode_categories['Parallel'].append(info.name)
            elif 'GAS' in info.name:
                opcode_categories['Gas Optimization'].append(info.name)
            elif 'BIGINT' in info.name or 'MATH' in info.name:
                opcode_categories['Math'].append(info.name)
            elif 'STRING' in info.name:
                opcode_categories['String'].append(info.name)
            elif 'ARRAY' in info.name:
                opcode_categories['Array'].append(info.name)
            elif 'JSON' in info.name:
                opcode_categories['JSON'].append(info.name)
            elif 'TIME' in info.name or 'BLOCK' in info.name:
                opcode_categories['Time'].append(info.name)
            elif 'RANDOM' in info.name:
                opcode_categories['Random'].append(info.name)
            elif 'DEBUG' in info.name:
                opcode_categories['Debug'].append(info.name)
        
        for category, opcodes in opcode_categories.items():
            if opcodes:
                print(f"  {category}: {len(opcodes)} opcodes")
                for opcode in opcodes[:3]:  # Show first 3
                    print(f"    - {opcode}")
                if len(opcodes) > 3:
                    print(f"    ... and {len(opcodes) - 3} more")
        
        # Demonstrate specific advanced opcodes
        print("\n🧪 Testing Advanced Opcodes:")
        
        # Test cryptographic opcodes
        print("  🔐 Cryptographic Operations:")
        crypto_opcodes = [
            AdvancedOpcodeEnum.KECCAK256,
            AdvancedOpcodeEnum.BLAKE2B,
            AdvancedOpcodeEnum.SHA3_256,
            AdvancedOpcodeEnum.ECDSA_VERIFY
        ]
        
        for opcode in crypto_opcodes:
            info = advanced_opcode_registry.get_opcode_info(opcode)
            if info:
                print(f"    ✅ {info.name}: {info.description} (Gas: {info.gas_cost})")
        
        # Test memory opcodes
        print("  💾 Memory Operations:")
        memory_opcodes = [
            AdvancedOpcodeEnum.MEMCOPY,
            AdvancedOpcodeEnum.MEMCMP,
            AdvancedOpcodeEnum.MEMSET,
            AdvancedOpcodeEnum.MEMFIND
        ]
        
        for opcode in memory_opcodes:
            info = advanced_opcode_registry.get_opcode_info(opcode)
            if info:
                print(f"    ✅ {info.name}: {info.description} (Gas: {info.gas_cost})")
        
        # Test parallel execution opcodes
        print("  ⚡ Parallel Execution:")
        parallel_opcodes = [
            AdvancedOpcodeEnum.PARALLEL_START,
            AdvancedOpcodeEnum.PARALLEL_END,
            AdvancedOpcodeEnum.PARALLEL_FORK,
            AdvancedOpcodeEnum.PARALLEL_JOIN
        ]
        
        for opcode in parallel_opcodes:
            info = advanced_opcode_registry.get_opcode_info(opcode)
            if info:
                print(f"    ✅ {info.name}: {info.description} (Gas: {info.gas_cost})")
    
    def demonstrate_execution_optimizations(self) -> None:
        """Demonstrate execution optimizations."""
        print("\n⚡ EXECUTION OPTIMIZATIONS DEMONSTRATION")
        print("=" * 50)
        
        # Test basic execution
        print("📊 Basic Execution Performance:")
        basic_times = []
        basic_gas_usage = []
        
        for i, contract in enumerate(self.demo_contracts):
            start_time = time.time()
            result = self.advanced_engine.execute_contract(
                contract=contract,
                caller='0x0000000000000000000000000000000000000000',
                value=0,
                data=b"test_input",
                gas_limit=1000000,
                block_context={'block_number': 1, 'timestamp': 1234567890}
            )
            execution_time = time.time() - start_time
            
            basic_times.append(execution_time)
            basic_gas_usage.append(result.gas_used)
            
            print(f"  Contract {i + 1}: {execution_time:.4f}s, {result.gas_used} gas")
        
        # Test optimized execution
        print("\n🚀 Optimized Execution Performance:")
        optimized_times = []
        optimized_gas_usage = []
        
        for i, contract in enumerate(self.demo_contracts):
            start_time = time.time()
            result = self.advanced_engine.execute_contract_advanced(
                contract, b"test_input", 1000000, 
                enable_optimizations=True, enable_caching=True
            )
            execution_time = time.time() - start_time
            
            optimized_times.append(execution_time)
            optimized_gas_usage.append(result.gas_used)
            
            print(f"  Contract {i + 1}: {execution_time:.4f}s, {result.gas_used} gas")
        
        # Calculate improvements
        print("\n📈 Performance Improvements:")
        for i in range(len(self.demo_contracts)):
            time_improvement = ((basic_times[i] - optimized_times[i]) / basic_times[i]) * 100
            gas_improvement = ((basic_gas_usage[i] - optimized_gas_usage[i]) / basic_gas_usage[i]) * 100
            
            print(f"  Contract {i + 1}:")
            print(f"    Time improvement: {time_improvement:.2f}%")
            print(f"    Gas improvement: {gas_improvement:.2f}%")
    
    def demonstrate_execution_caching(self) -> None:
        """Demonstrate execution caching."""
        print("\n💾 EXECUTION CACHING DEMONSTRATION")
        print("=" * 50)
        
        # Test without caching
        print("📊 Execution without caching:")
        start_time = time.time()
        for _ in range(5):
            result = self.advanced_engine.execute_contract_advanced(
                self.demo_contracts[0], b"test_input", 1000000,
                enable_optimizations=False, enable_caching=False
            )
        no_cache_time = time.time() - start_time
        print(f"  Total time: {no_cache_time:.4f}s")
        
        # Test with caching
        print("\n🚀 Execution with caching:")
        start_time = time.time()
        for _ in range(5):
            result = self.advanced_engine.execute_contract_advanced(
                self.demo_contracts[0], b"test_input", 1000000,
                enable_optimizations=False, enable_caching=True
            )
        with_cache_time = time.time() - start_time
        print(f"  Total time: {with_cache_time:.4f}s")
        
        # Show cache metrics
        cache_metrics = self.advanced_engine.get_cache_metrics()
        print(f"\n📈 Cache Metrics:")
        print(f"  Cache size: {cache_metrics['cache_size']}")
        print(f"  Hit rate: {cache_metrics['hit_rate']:.2%}")
        print(f"  Hit count: {cache_metrics['hit_count']}")
        print(f"  Miss count: {cache_metrics['miss_count']}")
        
        # Calculate improvement
        if no_cache_time > 0:
            improvement = ((no_cache_time - with_cache_time) / no_cache_time) * 100
            print(f"  Performance improvement: {improvement:.2f}%")
    
    def demonstrate_parallel_execution(self) -> None:
        """Demonstrate parallel execution."""
        print("\n⚡ PARALLEL EXECUTION DEMONSTRATION")
        print("=" * 50)
        
        # Create contracts with parallel execution markers
        parallel_contract = SmartContract(
            address="0x5555555555555555555555555555555555555555",
            bytecode=b"PARALLEL_START\x60\x01\x60\x02\x01PARALLEL_END",
            contract_type=ContractType.STANDARD,
            name="Parallel Contract",
            version="1.0.0"
        )
        
        print("📊 Sequential Execution:")
        start_time = time.time()
        for i in range(3):
            result = self.advanced_engine.execute_contract_advanced(
                parallel_contract, b"test_input", 1000000,
                enable_optimizations=False, enable_caching=False, enable_parallel=False
            )
        sequential_time = time.time() - start_time
        print(f"  Total time: {sequential_time:.4f}s")
        
        print("\n🚀 Parallel Execution:")
        start_time = time.time()
        for i in range(3):
            result = self.advanced_engine.execute_contract_advanced(
                parallel_contract, b"test_input", 1000000,
                enable_optimizations=False, enable_caching=False, enable_parallel=True
            )
        parallel_time = time.time() - start_time
        print(f"  Total time: {parallel_time:.4f}s")
        
        # Calculate improvement
        if sequential_time > 0:
            improvement = ((sequential_time - parallel_time) / sequential_time) * 100
            print(f"  Performance improvement: {improvement:.2f}%")
    
    def demonstrate_gas_optimization(self) -> None:
        """Demonstrate gas optimization."""
        print("\n⛽ GAS OPTIMIZATION DEMONSTRATION")
        print("=" * 50)
        
        # Test gas usage with different optimization levels
        print("📊 Gas Usage Analysis:")
        
        for i, contract in enumerate(self.demo_contracts):
            # Without optimizations
            result_basic = self.advanced_engine.execute_contract_advanced(
                contract, b"test_input", 1000000,
                enable_optimizations=False, enable_caching=False
            )
            
            # With optimizations
            result_optimized = self.advanced_engine.execute_contract_advanced(
                contract, b"test_input", 1000000,
                enable_optimizations=True, enable_caching=False
            )
            
            gas_saved = result_basic.gas_used - result_optimized.gas_used
            gas_improvement = (gas_saved / result_basic.gas_used) * 100 if result_basic.gas_used > 0 else 0
            
            print(f"  Contract {i + 1}:")
            print(f"    Basic gas usage: {result_basic.gas_used}")
            print(f"    Optimized gas usage: {result_optimized.gas_used}")
            print(f"    Gas saved: {gas_saved} ({gas_improvement:.2f}%)")
    
    def show_vm_metrics(self) -> None:
        """Show comprehensive VM metrics."""
        print("\n📊 ADVANCED VM METRICS")
        print("=" * 50)
        
        # Execution metrics
        execution_metrics = self.advanced_engine.get_execution_metrics()
        print("⚙️  Execution Metrics:")
        print(f"  Total executions: {execution_metrics.total_executions}")
        print(f"  Successful executions: {execution_metrics.successful_executions}")
        print(f"  Failed executions: {execution_metrics.failed_executions}")
        print(f"  Success rate: {execution_metrics.success_rate:.2%}")
        print(f"  Average execution time: {execution_metrics.average_execution_time:.4f}s")
        print(f"  Average gas used: {execution_metrics.average_gas_used}")
        print(f"  Cache hits: {execution_metrics.cache_hits}")
        print(f"  Cache misses: {execution_metrics.cache_misses}")
        print(f"  Cache hit rate: {execution_metrics.cache_hit_rate:.2%}")
        print(f"  Parallel executions: {execution_metrics.parallel_executions}")
        print(f"  Optimizations applied: {execution_metrics.optimization_applied}")
        
        # Cache metrics
        cache_metrics = self.advanced_engine.get_cache_metrics()
        print("\n💾 Cache Metrics:")
        print(f"  Cache size: {cache_metrics['cache_size']}")
        print(f"  Max cache size: {cache_metrics['max_size']}")
        print(f"  Hit rate: {cache_metrics['hit_rate']:.2%}")
        print(f"  Total cache requests: {cache_metrics['hit_count'] + cache_metrics['miss_count']}")
        
        # Performance monitor metrics
        performance_stats = self.advanced_engine.performance_monitor.get_all_performance_stats()
        if performance_stats:
            print("\n📈 Performance Statistics:")
            for operation, stats in performance_stats.items():
                if stats:
                    print(f"  {operation}:")
                    print(f"    Count: {stats['count']}")
                    print(f"    Average: {stats['avg']:.4f}")
                    print(f"    Min: {stats['min']:.4f}")
                    print(f"    Max: {stats['max']:.4f}")
    
    def run_demo(self) -> None:
        """Run the complete advanced VM demo."""
        print("🚀 DUBCHAIN ADVANCED VIRTUAL MACHINE DEMO")
        print("=" * 60)
        print("This demo showcases enterprise-grade VM enhancements")
        print("including advanced opcodes, optimizations, parallel")
        print("execution, and performance monitoring.")
        print("=" * 60)
        
        # Setup
        self.setup_demo_contracts()
        
        # Demonstrate features
        self.demonstrate_advanced_opcodes()
        self.demonstrate_execution_optimizations()
        self.demonstrate_execution_caching()
        self.demonstrate_parallel_execution()
        self.demonstrate_gas_optimization()
        self.show_vm_metrics()
        
        print("\n🎉 DEMO COMPLETED!")
        print("=" * 60)
        print("DubChain's advanced VM provides:")
        print("✅ Advanced opcodes and operations")
        print("✅ Just-in-time compilation")
        print("✅ Parallel execution support")
        print("✅ Advanced gas optimization")
        print("✅ Performance monitoring")
        print("✅ Execution caching")
        print("✅ Enterprise-grade performance")
        print("=" * 60)


async def main():
    """Main demo function."""
    demo = AdvancedVMDemo()
    demo.run_demo()


if __name__ == "__main__":
    asyncio.run(main())
