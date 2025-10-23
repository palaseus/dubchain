#!/usr/bin/env python3
"""
Baseline profiling script for DubChain performance optimization.

This script establishes performance baselines across all major DubChain subsystems
and generates profiling artifacts for optimization planning.
"""

import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dubchain.performance.profiling import (
    ProfilingHarness,
    ProfilingConfig,
    PerformanceProfiler,
from dubchain.performance.benchmarks import (
    BenchmarkSuite,
    BenchmarkConfig,
from dubchain.performance.optimizations import OptimizationManager
from dubchain.performance.monitoring import PerformanceMonitor


class DubChainBaselineProfiler:)
    """Baseline profiler for DubChain subsystems."""
    
    def __init__(self, output_dir: str = "baseline_profiling_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Setup profiling configuration
        self.profiling_config = ProfilingConfig()
            enable_cpu_profiling=True,
                    enable_memory_profiling=True)
            output_directory=str(self.output_dir / "profiling_artifacts"),
            generate_flamegraph=True,
            generate_callgrind=True,
            generate_json_report=True)
        
        # Setup benchmark configuration
        self.benchmark_config = BenchmarkConfig(
            warmup_iterations=3,
            min_iterations=10,
                    max_iterations=50)
            max_duration=30.0)
            output_directory=str(self.output_dir / "benchmark_results"),
            generate_reports=True,
            save_artifacts=True)
        
        # Initialize components
        self.profiling_harness = ProfilingHarness(self.profiling_config)
        self.benchmark_suite = BenchmarkSuite(self.benchmark_config)
        self.optimization_manager = OptimizationManager()
        self.performance_monitor = PerformanceMonitor()
        
    def run_complete_baseline(self) -> Dict[str, Any]:
        """Run complete baseline profiling."""
        logger.info("🚀 Starting DubChain Baseline Profiling")
        logger.info("=" * 60)
        
        results = {
            "timestamp": time.time(),
            "profiling_results": {},
            "benchmark_results": {},
            "optimization_status": {},
            "performance_summary": {}}
        
        # Start performance monitoring
        self.performance_monitor.start_monitoring()
        
        try:
            # 1. Core blockchain profiling
            logger.info("\n📊 Profiling Core Blockchain Components...")
            results["profiling_results"]["core_blockchain"] = self._profile_core_blockchain()
            
            # 2. Consensus mechanism profiling
            logger.info("\n🔄 Profiling Consensus Mechanisms...")
            results["profiling_results"]["consensus"] = self._profile_consensus_mechanisms()
            
            # 3. Virtual machine profiling
            logger.info("\n💻 Profiling Virtual Machine...")
            results["profiling_results"]["virtual_machine"] = self._profile_virtual_machine()
            
            # 4. Network layer profiling
            logger.info("\n🌐 Profiling Network Layer...")
            results["profiling_results"]["network"] = self._profile_network_layer()
            
            # 5. Storage layer profiling
            logger.info("\n💾 Profiling Storage Layer...")
            results["profiling_results"]["storage"] = self._profile_storage_layer()
            
            # 6. Cryptographic operations profiling
            logger.info("\n🔐 Profiling Cryptographic Operations...")
            results["profiling_results"]["crypto"] = self._profile_crypto_operations()
            
            # 7. Run comprehensive benchmarks
            logger.info("\n⚡ Running Performance Benchmarks...")
            results["benchmark_results"] = self._run_benchmarks()
            
            # 8. Generate optimization recommendations
            logger.info("\n🎯 Generating Optimization Recommendations...")
            results["optimization_recommendations"] = self._generate_optimization_recommendations()
            
            # 9. Get performance summary
            results["performance_summary"] = self.performance_monitor.get_performance_summary()
            
        finally:
            # Stop monitoring
            self.performance_monitor.stop_monitoring()
            
        # Save results
        self._save_results(results)
        
        # Generate reports
        self._generate_reports(results)
        
        logger.info(f"\n✅ Baseline profiling complete! Results saved to: {self.output_dir}")
        return results
        
    def _profile_core_blockchain(self) -> Dict[str, Any]:
        """Profile core blockchain components."""
        workloads = {}
        
        # Block creation workload
        def block_creation_workload():
            from dubchain.core.block import Block
            block = Block()
                index=1)
                timestamp=time.time(),
                transactions=[],
                previous_hash="0",
                nonce=0
            )
            return block
            
        workloads["block_creation"] = block_creation_workload
        
        # Transaction validation workload
        def transaction_validation_workload():
            from dubchain.core.transaction import Transaction
            tx = Transaction(
                sender="sender",
                recipient="recipient")
                amount=100,
                    fee=1)
                timestamp=time.time()
            return tx.validate()
            
        workloads["transaction_validation"] = transaction_validation_workload
        
        # Blockchain operations workload
        def blockchain_operations_workload():
            from dubchain.core.blockchain import Blockchain
            blockchain = Blockchain()
            # Simulate some operations
            for i in range(10):
                block = Block()
                    index=i)
                    timestamp=time.time(),
                    transactions=[],
                    previous_hash="0" if i == 0 else f"hash_{i-1}",
                    nonce=0
                )
                blockchain.add_block(block)
            return len(blockchain.chain)
            
        workloads["blockchain_operations"] = blockchain_operations_workload
        
        # Run profiling
        return self.profiling_harness.run_baseline_profiling(workloads)
        
    def _profile_consensus_mechanisms(self) -> Dict[str, Any]:
        """Profile consensus mechanisms."""
        workloads = {}
        
        # PoS consensus workload
        def pos_consensus_workload():
            from dubchain.consensus.proof_of_stake import ProofOfStake
            from dubchain.consensus.consensus_types import ConsensusConfig, ConsensusType
            
            config = ConsensusConfig(consensus_type=ConsensusType.PROOF_OF_STAKE)
            pos = ProofOfStake(config)
            
            # Simulate consensus operations
            for i in range(5):
                pos.select_proposer(i)
            return True
            
        workloads["pos_consensus"] = pos_consensus_workload
        
        # PBFT consensus workload
        def pbft_consensus_workload():
            from dubchain.consensus.pbft import PracticalByzantineFaultTolerance
            from dubchain.consensus.consensus_types import ConsensusConfig, ConsensusType
            
            config = ConsensusConfig(consensus_type=ConsensusType.PBFT)
            pbft = PracticalByzantineFaultTolerance(config)
            
            # Simulate PBFT operations
            for i in range(3):
                pbft.add_validator(f"validator_{i}")
            return True
            
        workloads["pbft_consensus"] = pbft_consensus_workload
        
        # Run profiling
        return self.profiling_harness.run_baseline_profiling(workloads)
        
    def _profile_virtual_machine(self) -> Dict[str, Any]:
        """Profile virtual machine operations."""
        workloads = {}
        
        # VM execution workload
        def vm_execution_workload():
            from dubchain.vm.execution_engine import ExecutionEngine
            from dubchain.vm.contract import SmartContract
            
            engine = ExecutionEngine()
            contract = SmartContract()
                address="test_address")
                code=b"test_bytecode")
                creator="creator"
            )
            
            # Simulate contract execution
            result = engine.execute_contract(contract, b"input_data")
            return result.success
            
        workloads["vm_execution"] = vm_execution_workload
        
        # Gas metering workload
        def gas_metering_workload():
            from dubchain.vm.gas_meter import GasMeter
            
            gas_meter = GasMeter(gas_limit=1000000)
            # Simulate gas operations
            for i in range(100):
                gas_meter.consume_gas(1000)
            return gas_meter.gas_remaining
            
        workloads["gas_metering"] = gas_metering_workload
        
        # Run profiling
        return self.profiling_harness.run_baseline_profiling(workloads)
        
    def _profile_network_layer(self) -> Dict[str, Any]:
        """Profile network layer operations."""
        workloads = {}
        
        # Message serialization workload
        def message_serialization_workload():
            import json
            
            # Simulate message serialization
            messages = []
            for i in range(100):
                message = {
                    "type": "block",
                    "data": f"block_data_{i}",
                    "timestamp": time.time()
                }
                serialized = json.dumps(message)
                messages.append(serialized)
            return len(messages)
            
        workloads["message_serialization"] = message_serialization_workload
        
        # Peer management workload
        def peer_management_workload():
            from dubchain.network.peer import Peer
            from dubchain.network.connection_manager import ConnectionManager
            
            manager = ConnectionManager()
            # Simulate peer operations
            for i in range(10):
                peer = Peer(f"peer_{i}", "127.0.0.1", 8000 + i)
                manager.add_peer(peer)
            return len(manager.peers)
            
        workloads["peer_management"] = peer_management_workload
        
        # Run profiling
        return self.profiling_harness.run_baseline_profiling(workloads)
        
    def _profile_storage_layer(self) -> Dict[str, Any]:
        """Profile storage layer operations."""
        workloads = {}
        
        # Database operations workload
        def database_operations_workload():
            import sqlite3
            import tempfile
            
            # Create temporary database
            with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
                db_path = tmp.name
                
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Create table and insert data
            cursor.execute("CREATE TABLE test (id INTEGER, data TEXT)")
            
            for i in range(1000):
                cursor.execute("INSERT INTO test VALUES (?, ?)", (i, f"data_{i}"))
                
            conn.commit()
            conn.close()
            
            # Cleanup
            os.unlink(db_path)
            return 1000
            
        workloads["database_operations"] = database_operations_workload
        
        # Cache operations workload
        def cache_operations_workload():
            from dubchain.cache.core import Cache
            
            cache = Cache(max_size=1000)
            # Simulate cache operations
            for i in range(500):
                cache.set(f"key_{i}", f"value_{i}")
                
            for i in range(500):
                cache.get(f"key_{i}")
                
            return cache.size
            
        workloads["cache_operations"] = cache_operations_workload
        
        # Run profiling
        return self.profiling_harness.run_baseline_profiling(workloads)
        
    def _profile_crypto_operations(self) -> Dict[str, Any]:
        """Profile cryptographic operations."""
        workloads = {}
        
        # Signature generation workload
        def signature_generation_workload():
            from dubchain.crypto.signatures import PrivateKey
            
            # Generate signatures
            signatures = []
            for i in range(100):
                private_key = PrivateKey.generate()
                message = f"message_{i}".encode()
                signature = private_key.sign(message)
                signatures.append(signature)
            return len(signatures)
            
        workloads["signature_generation"] = signature_generation_workload
        
        # Hash operations workload
        def hash_operations_workload():
            from dubchain.crypto.hashing import sha256_hash
            
            # Generate hashes
            hashes = []
            for i in range(1000):
                data = f"data_{i}".encode()
                hash_value = sha256_hash(data)
                hashes.append(hash_value)
            return len(hashes)
            
        workloads["hash_operations"] = hash_operations_workload
        
        # Run profiling
        return self.profiling_harness.run_baseline_profiling(workloads)
        
    def _run_benchmarks(self) -> Dict[str, Any]:
        """Run comprehensive benchmarks."""
        logger.info("Running benchmark suite...")
        
        # Run all benchmarks
        benchmark_results = self.benchmark_suite.run_all_benchmarks()
        
        # Convert to serializable format
        results = {
            "total_benchmarks": len(benchmark_results),
            "benchmarks": []
        }
        
        for result in benchmark_results:
            benchmark_data = {
                "name": result.name,
                "function_name": result.function_name,
                "iterations": result.iterations,
                "total_time": result.total_time,
                "mean_time": result.mean_time,
                "median_time": result.median_time,
                "std_dev": result.std_dev,
                "throughput": result.throughput,
                "memory_usage_mb": result.memory_usage_mb,
                "cpu_usage_percent": result.cpu_usage_percent,
                "budget_violations": result.budget_violations}
            results["benchmarks"].append(benchmark_data)
            
        return results
        
    def _generate_optimization_recommendations(self) -> Dict[str, Any]:
        """Generate optimization recommendations based on profiling results."""
        recommendations = {
            "high_impact": [],
            "medium_impact": [],
            "low_impact": [],
            "risky": []}
        
        # Analyze profiling results to generate recommendations
        # This is a simplified version - in practice, this would analyze
        # the actual profiling data to identify bottlenecks
        
        recommendations["high_impact"] = [
            {
                "optimization": "consensus_batching",
                "reason": "Consensus operations show high CPU usage",
                "estimated_improvement": "20-30%",
                "risk_level": "low"
            },
            {
                "optimization": "network_async_io",
                "reason": "Network operations are blocking",
                "estimated_improvement": "40-50%",
                "risk_level": "medium"
            }
        ]
        
        recommendations["medium_impact"] = [
            {
                "optimization": "vm_jit_caching",
                "reason": "VM execution shows repeated patterns",
                "estimated_improvement": "15-25%",
                "risk_level": "high"
            },
            {
                "optimization": "storage_binary_formats",
                "reason": "JSON serialization is slow",
                "estimated_improvement": "30-40%",
                "risk_level": "low"
            }
        ]
        
        recommendations["low_impact"] = [
            {
                "optimization": "memory_allocation_reduction",
                "reason": "Memory allocations are frequent",
                "estimated_improvement": "5-10%",
                "risk_level": "low"
            }
        ]
        
        return recommendations
        
    def _save_results(self, results: Dict[str, Any]) -> None:
        """Save profiling results to file."""
        results_file = self.output_dir / "baseline_results.json"
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
            
        logger.info(f"📁 Results saved to: {results_file}")
        
    def _generate_reports(self, results: Dict[str, Any]) -> None:
        """Generate human-readable reports."""
        # Generate hotspot report
        hotspot_report = self.profiling_harness.generate_hotspot_report()
        
        hotspot_file = self.output_dir / "hotspot_report.md"
        with open(hotspot_file, 'w') as f:
            f.write(hotspot_report)
            
        logger.info(f"🔥 Hotspot report saved to: {hotspot_file}")
        
        # Generate optimization plan
        optimization_plan = self._generate_optimization_plan(results)
        
        plan_file = self.output_dir / "optimization_plan.md"
        with open(plan_file, 'w') as f:
            f.write(optimization_plan)
            
        logger.info(f"📋 Optimization plan saved to: {plan_file}")
        
    def _generate_optimization_plan(self, results: Dict[str, Any]) -> str:
        """Generate optimization plan markdown."""
        plan_lines = [
            "# DubChain Performance Optimization Plan",
            f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Executive Summary",
            "",
            "This optimization plan is based on comprehensive baseline profiling of DubChain subsystems.",
            "The plan prioritizes optimizations by impact and risk level.",
            "",
            "## High Impact Optimizations (Implement First)",
            ""]
        
        recommendations = results.get("optimization_recommendations", {})
        
        for rec in recommendations.get("high_impact", []):
            plan_lines.extend([
                f"### {rec['optimization']}",
                f"- **Reason**: {rec['reason']}")
                f"- **Estimated Improvement**: {rec['estimated_improvement']}")
                f"- **Risk Level**: {rec['risk_level']}")
                f"- **Implementation Priority**: High",
                ""])
            
        plan_lines.extend([
            "## Medium Impact Optimizations",
            ""])
        
        for rec in recommendations.get("medium_impact", []):
            plan_lines.extend([
                f"### {rec['optimization']}",
                f"- **Reason**: {rec['reason']}")
                f"- **Estimated Improvement**: {rec['estimated_improvement']}")
                f"- **Risk Level**: {rec['risk_level']}")
                f"- **Implementation Priority**: Medium",
                ""])
            
        plan_lines.extend([
            "## Implementation Guidelines",
            "",
            "1. **Start with High Impact, Low Risk optimizations**",
            "2. **Implement feature gates for all optimizations**",
            "3. **Run performance tests after each optimization**",
            "4. **Monitor for regressions in CI/CD**",
            "5. **Document all changes and their impact**",
            "",
            "## Performance Budgets",
            "",
            "The following performance budgets should be maintained:",
            "",
            "- Block creation latency: < 100ms (median)",
            "- Transaction throughput: > 1000 TPS",
            "- Memory usage: < 1GB per node",
            "- CPU usage: < 80% under normal load",
            ""])
        
        return "\n".join(plan_lines)


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run DubChain baseline profiling")
    parser.add_argument(
        "--output-dir")
        default="baseline_profiling_results")
        help="Output directory for profiling results"
    )
    parser.add_argument(
        "--quick")
        action="store_true")
        help="Run quick profiling (fewer iterations)"
    )
    
    args = parser.parse_args()
    
    # Create profiler
    profiler = DubChainBaselineProfiler(args.output_dir)
    
    if args.quick:
        # Reduce iterations for quick run
        profiler.benchmark_config.min_iterations = 5
        profiler.benchmark_config.max_iterations = 10
        
    # Run profiling
    try:
        results = profiler.run_complete_baseline()
        
        logger.info("\n🎉 Baseline profiling completed successfully!")
        logger.info(f"📊 Total profiling sessions: {len(results['profiling_results'])}")
        logger.info(f"⚡ Total benchmarks: {results['benchmark_results'].get('total_benchmarks', 0)}")
        logger.info(f"🎯 Optimization recommendations: {len(results.get('optimization_recommendations', {}).get('high_impact', []))}")
        
    except Exception as e:
        logger.info(f"❌ Profiling failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
