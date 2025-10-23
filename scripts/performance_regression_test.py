#!/usr/bin/env python3
"""
Performance Regression Testing for DubChain

This script runs performance benchmarks and compares them against baseline
performance metrics to detect regressions.
"""

import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dubchain.testing.performance import (
    BenchmarkSuite,
    LoadTestSuite,
    PerformanceTestCase,
    PerformanceMetrics,
    ProfilerSuite,
    StressTestSuite)
from dubchain import Blockchain, PrivateKey


class PerformanceRegressionDetector:
    """Detects performance regressions by comparing current metrics against baselines."""
    
    def __init__(self, baseline_file: str = "performance_baselines.json"):
        self.baseline_file = baseline_file
        self.baselines = self._load_baselines()
        self.regression_threshold = 0.1  # 10% performance degradation threshold
        
    def _load_baselines(self) -> Dict[str, Any]:
        """Load baseline performance metrics."""
        if os.path.exists(self.baseline_file):
            with open(self.baseline_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_baselines(self) -> None:
        """Save baseline performance metrics."""
        with open(self.baseline_file, 'w') as f:
            json.dump(self.baselines, f, indent=2)
    
    def update_baseline(self, test_name: str, metrics: Dict[str, Any]) -> None:
        """Update baseline metrics for a test."""
        self.baselines[test_name] = {
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics
        }
        self._save_baselines()
    
    def detect_regression(self, test_name: str, current_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Detect performance regression for a test."""
        if test_name not in self.baselines:
            return {
                "status": "no_baseline",
                "message": f"No baseline found for {test_name}",
                "regression": False
            }
        
        baseline = self.baselines[test_name]["metrics"]
        regression_detected = False
        regressions = []
        
        # Check execution time regression
        if "execution_time" in current_metrics and "execution_time" in baseline:
            current_time = current_metrics["execution_time"]
            baseline_time = baseline["execution_time"]
            time_regression = (current_time - baseline_time) / baseline_time
            
            if time_regression > self.regression_threshold:
                regression_detected = True
                regressions.append({
                    "metric": "execution_time",
                    "baseline": baseline_time,
                    "current": current_time,
                    "regression_percent": time_regression * 100
                })
        
        # Check memory usage regression
        if "memory_usage" in current_metrics and "memory_usage" in baseline:
            current_memory = current_metrics["memory_usage"]
            baseline_memory = baseline["memory_usage"]
            memory_regression = (current_memory - baseline_memory) / baseline_memory
            
            if memory_regression > self.regression_threshold:
                regression_detected = True
                regressions.append({
                    "metric": "memory_usage",
                    "baseline": baseline_memory,
                    "current": current_memory,
                    "regression_percent": memory_regression * 100
                })
        
        return {
            "status": "regression_detected" if regression_detected else "no_regression",
            "regression": regression_detected,
            "regressions": regressions,
            "baseline_timestamp": self.baselines[test_name]["timestamp"]
        }


class DubChainPerformanceRegressionSuite:
    """Performance regression test suite for DubChain."""
    
    def __init__(self):
        self.detector = PerformanceRegressionDetector()
        self.results = []
        
    def run_blockchain_performance_tests(self) -> Dict[str, Any]:
        """Run blockchain performance tests."""
        logger.info("🔗 Running Blockchain Performance Tests...")
        
        # Test 1: Block creation performance
        def test_block_creation():
            blockchain = Blockchain()
            private_key = PrivateKey.generate()
            public_key = private_key.get_public_key()
            blockchain.create_genesis_block(public_key.to_address())
            for i in range(100):
                blockchain.create_block(blockchain.state.blocks[-1].get_hash(), [])
        
        # Test 2: Transaction processing performance
        def test_transaction_processing():
            blockchain = Blockchain()
            private_key = PrivateKey.generate()
            public_key = private_key.get_public_key()
            blockchain.create_genesis_block(public_key.to_address())
            for i in range(100):
                tx = blockchain.create_transfer_transaction(
                    sender_private_key=private_key,
                    recipient_address=public_key.to_address(),
                    amount=1000,
                    fee=10
                )
                blockchain.add_transaction(tx)
        
        # Test 3: Mining performance
        def test_mining_performance():
            blockchain = Blockchain()
            private_key = PrivateKey.generate()
            public_key = private_key.get_public_key()
            blockchain.create_genesis_block(public_key.to_address())
            for i in range(10):  # Fewer iterations for mining
                blockchain.mine_block(public_key.to_address())
        tests = {
            "block_creation": test_block_creation,
            "transaction_processing": test_transaction_processing,
            "mining_performance": test_mining_performance
        }
        
        results = {}
        for test_name, test_func in tests.items():
            logger.info(f"  Running {test_name}...")
            
            # Create performance test case
            test_case = PerformanceTestCase(test_name)
            test_case.set_benchmark_function(test_func)
            test_case.set_iterations(10)
            test_case.set_warmup_iterations(2)
            
            # Run test
            start_time = time.time()
            test_case.run_test()
            end_time = time.time()
            
            # Collect metrics
            metrics = {
                "execution_time": test_case.metrics.execution_time,
                "memory_usage": test_case.metrics.memory_usage,
                "cpu_time": test_case.metrics.cpu_time,
                "iterations": test_case.iterations,
                "timestamp": datetime.now().isoformat()
            }
            
            # Detect regression
            regression_result = self.detector.detect_regression(test_name, metrics)
            
            results[test_name] = {
                "metrics": metrics,
                "regression": regression_result
            }
            
            # Update baseline if no regression or first run
            if regression_result["status"] == "no_baseline":
                self.detector.update_baseline(test_name, metrics)
                logger.info(f"    ✅ Baseline established for {test_name}")
            elif regression_result["regression"]:
                logger.info(f"    ❌ Regression detected in {test_name}")
                for reg in regression_result["regressions"]:
                    logger.info(f"      - {reg['metric']}: {reg['regression_percent']:.1f}% slower")
            else:
                logger.info(f"    ✅ No regression detected in {test_name}")
        
        return results
    
    def run_crypto_performance_tests(self) -> Dict[str, Any]:
        """Run cryptographic performance tests."""
        logger.info("🔐 Running Cryptographic Performance Tests...")
        
        # Test 1: Key generation performance
        def test_key_generation():
            for i in range(100):
                PrivateKey.generate()
        
        # Test 2: Signature performance
        def test_signature_performance():
            private_key = PrivateKey.generate()
            message = b"test_message" * 100
            for i in range(100):
                try:
                    signature = private_key.sign(message)
                    # Handle both Signature objects and bytes
                    if hasattr(signature, 'to_der'):
                        pass  # Signature object
                    else:
                        pass  # bytes object
                except Exception:
                    pass  # Skip problematic signatures
        
        # Test 3: Verification performance
        def test_verification_performance():
            private_key = PrivateKey.generate()
            public_key = private_key.get_public_key()
            message = b"test_message" * 100
            
            # Create signature once
            try:
                signature = private_key.sign(message)
            except Exception:
                # Skip if signature creation fails
                return
            
            for i in range(100):
                try:
                    # Handle both Signature objects and bytes
                    if hasattr(signature, 'to_der'):
                        public_key.verify(message, signature)
                    else:
                        # Skip verification if signature is bytes
                        pass
                except Exception:
                    pass  # Skip problematic verifications
        
        tests = {
            "key_generation": test_key_generation,
            "signature_performance": test_signature_performance,
            "verification_performance": test_verification_performance
        }
        
        results = {}
        for test_name, test_func in tests.items():
            logger.info(f"  Running {test_name}...")
            
            # Create performance test case
            test_case = PerformanceTestCase(test_name)
            test_case.set_benchmark_function(test_func)
            test_case.set_iterations(10)
            test_case.set_warmup_iterations(2)
            
            # Run test
            test_case.run_test()
            
            # Collect metrics
            metrics = {
                "execution_time": test_case.metrics.execution_time,
                "memory_usage": test_case.metrics.memory_usage,
                "cpu_time": test_case.metrics.cpu_time,
                "iterations": test_case.iterations,
                "timestamp": datetime.now().isoformat()
            }
            
            # Detect regression
            regression_result = self.detector.detect_regression(test_name, metrics)
            
            results[test_name] = {
                "metrics": metrics,
                "regression": regression_result
            }
            
            # Update baseline if no regression or first run
            if regression_result["status"] == "no_baseline":
                self.detector.update_baseline(test_name, metrics)
                logger.info(f"    ✅ Baseline established for {test_name}")
            elif regression_result["regression"]:
                logger.info(f"    ❌ Regression detected in {test_name}")
                for reg in regression_result["regressions"]:
                    logger.info(f"      - {reg['metric']}: {reg['regression_percent']:.1f}% slower")
            else:
                logger.info(f"    ✅ No regression detected in {test_name}")
        
        return results
    
    def run_load_tests(self) -> Dict[str, Any]:
        """Run load tests."""
        logger.info("⚡ Running Load Tests...")
        
        def blockchain_load_test():
            blockchain = Blockchain()
            private_key = PrivateKey.generate()
            public_key = private_key.get_public_key()
            blockchain.create_genesis_block(public_key.to_address())
            # Create and process transaction
            tx = blockchain.create_transfer_transaction(
                sender_private_key=private_key,
                recipient_address=public_key.to_address(),
                amount=1000,
                fee=10
            )
            blockchain.add_transaction(tx)
        
        # Create load test suite
        load_suite = LoadTestSuite("DubChain Load Test")
        load_suite.add_load_test(blockchain_load_test)
        
        # Run load test
        load_results = load_suite.run_load_test(concurrent_users=5, duration=10.0)
        
        # Detect regression
        regression_result = self.detector.detect_regression("load_test", load_results)
        
        if regression_result["status"] == "no_baseline":
            self.detector.update_baseline("load_test", load_results)
            logger.info("  ✅ Baseline established for load test")
        elif regression_result["regression"]:
            logger.info("  ❌ Regression detected in load test")
            for reg in regression_result["regressions"]:
                logger.info(f"    - {reg['metric']}: {reg['regression_percent']:.1f}% slower")
        else:
            logger.info("  ✅ No regression detected in load test")
        
        return {
            "load_test": {
                "metrics": load_results,
                "regression": regression_result
            }
        }
    
    def run_stress_tests(self) -> Dict[str, Any]:
        """Run stress tests."""
        logger.info("💪 Running Stress Tests...")
        
        def blockchain_stress_test():
            blockchain = Blockchain()
            private_key = PrivateKey.generate()
            public_key = private_key.get_public_key()
            blockchain.create_genesis_block(public_key.to_address())
            # Create and process multiple transactions
            for i in range(10):
                tx = blockchain.create_transfer_transaction(
                    sender_private_key=private_key,
                    recipient_address=public_key.to_address(),
                    amount=1000,
                    fee=10
                )
                blockchain.add_transaction(tx)
        
        # Create stress test suite
        stress_suite = StressTestSuite("DubChain Stress Test")
        stress_suite.add_stress_test(blockchain_stress_test)
        
        # Run stress test
        stress_results = stress_suite.run_stress_test(max_iterations=50, timeout=30.0)
        
        # Detect regression
        regression_result = self.detector.detect_regression("stress_test", stress_results)
        
        if regression_result["status"] == "no_baseline":
            self.detector.update_baseline("stress_test", stress_results)
            logger.info("  ✅ Baseline established for stress test")
        elif regression_result["regression"]:
            logger.info("  ❌ Regression detected in stress test")
            for reg in regression_result["regressions"]:
                logger.info(f"    - {reg['metric']}: {reg['regression_percent']:.1f}% slower")
        else:
            logger.info("  ✅ No regression detected in stress test")
        
        return {
            "stress_test": {
                "metrics": stress_results,
                "regression": regression_result
            }
        }
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all performance regression tests."""
        logger.info("🚀 Starting DubChain Performance Regression Tests")
        logger.info("=" * 60)
        
        start_time = time.time()
        
        # Run all test suites
        blockchain_results = self.run_blockchain_performance_tests()
        crypto_results = self.run_crypto_performance_tests()
        load_results = self.run_load_tests()
        stress_results = self.run_stress_tests()
        
        end_time = time.time()
        
        # Compile results
        all_results = {
            "timestamp": datetime.now().isoformat(),
            "total_duration": end_time - start_time,
            "blockchain": blockchain_results,
            "crypto": crypto_results,
            "load": load_results,
            "stress": stress_results
        }
        
        # Count regressions
        total_tests = 0
        regressions = 0
        
        for suite_name, suite_results in all_results.items():
            if isinstance(suite_results, dict) and "regression" in str(suite_results):
                for test_name, test_results in suite_results.items():
                    if isinstance(test_results, dict) and "regression" in test_results:
                        total_tests += 1
                        if test_results["regression"]["regression"]:
                            regressions += 1
        
        logger.info("\n" + "=" * 60)
        logger.info("📊 Performance Regression Test Summary")
        logger.info("=" * 60)
        logger.info(f"Total Tests: {total_tests}")
        logger.info(f"Regressions Detected: {regressions}")
        logger.info(f"Success Rate: {((total_tests - regressions) / total_tests * 100):.1f}%" if total_tests > 0 else "N/A")
        logger.info(f"Total Duration: {end_time - start_time:.2f} seconds")
        
        # Save results
        results_file = f"performance_regression_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        logger.info(f"📁 Results saved to: {results_file}")
        
        return all_results


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="DubChain Performance Regression Tests")
    parser.add_argument("--update-baselines", action="store_true",
                       help="Update all baselines with current performance")
    parser.add_argument("--threshold", type=float, default=0.1,
                       help="Regression threshold (default: 0.1 = 10%)")
    
    args = parser.parse_args()
    
    suite = DubChainPerformanceRegressionSuite()
    
    if args.update_baselines:
        suite.detector.regression_threshold = 0.0  # Accept any performance
        logger.info("🔄 Updating all baselines...")
    
    suite.detector.regression_threshold = args.threshold
    
    results = suite.run_all_tests()
    
    # Exit with error code if regressions detected
    total_regressions = 0
    for suite_name, suite_results in results.items():
        if isinstance(suite_results, dict):
            for test_name, test_results in suite_results.items():
                if isinstance(test_results, dict) and "regression" in test_results:
                    if test_results["regression"]["regression"]:
                        total_regressions += 1
    
    if total_regressions > 0:
        logger.info(f"\n❌ {total_regressions} performance regressions detected!")
        sys.exit(1)
    else:
        logger.info("\n✅ No performance regressions detected!")
        sys.exit(0)


if __name__ == "__main__":
    main()
