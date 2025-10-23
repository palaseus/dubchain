#!/usr/bin/env python3
"""
DubChain Performance Stress Test

This script performs comprehensive performance testing of DubChain:
- Transaction throughput testing
- Block mining performance
- Memory usage analysis
- CPU utilization monitoring
- Network performance testing
- Storage I/O performance
- Concurrent operation testing
"""

import asyncio
import json
import os
import sys
import time
import threading
import psutil
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
import random
import statistics

logger = logging.getLogger(__name__)

import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dubchain import Blockchain, PrivateKey, PublicKey
from dubchain.core.consensus import ConsensusConfig


@dataclass
class PerformanceMetric:
    """Performance metric data structure."""
    name: str
    value: float
    unit: str
    timestamp: float
    details: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PerformanceTest:
    """Performance test data structure."""
    name: str
    duration: float
    metrics: List[PerformanceMetric] = field(default_factory=list)
    success: bool = True
    error: Optional[str] = None


class PerformanceStressTester:
    """Performance stress tester for DubChain."""
    
    def __init__(self, output_dir: str = "performance_stress_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.blockchain = None
        self.wallets = {}
        self.tests: List[PerformanceTest] = []
        
        # Performance monitoring
        self.monitoring = False
        self.monitoring_thread = None
        self.system_metrics = []
        
    def run_all_performance_tests(self) -> Dict[str, Any]:
        """Run all performance stress tests."""
        logger.info("⚡ Starting DubChain Performance Stress Tests")
        logger.info("=" * 80)
        
        start_time = time.time()
        
        try:
            # Initialize test environment
            self._initialize_test_environment()
            
            # Start system monitoring
            self._start_system_monitoring()
            
            # Run performance tests
            self._test_transaction_throughput()
            self._test_block_mining_performance()
            self._test_memory_performance()
            self._test_cpu_performance()
            self._test_concurrent_operations()
            self._test_storage_performance()
            self._test_network_performance()
            self._test_scalability()
            
            # Stop monitoring
            self._stop_system_monitoring()
            
            # Generate reports
            self._generate_performance_report()
            
        except Exception as e:
            logger.info(f"❌ Performance testing failed: {e}")
            import traceback
            traceback.print_exc()
            
        finally:
            total_duration = time.time() - start_time
            logger.info(f"\n✅ Performance testing completed in {total_duration:.2f} seconds")
            
        return self._get_performance_summary()
        
    def _initialize_test_environment(self):
        """Initialize test environment."""
        logger.info("\n🔧 Initializing performance test environment...")
        
        # Create blockchain with performance-optimized settings
        config = ConsensusConfig(
            target_block_time=0.5,  # Fast blocks for testing)
            difficulty_adjustment_interval=5,
                    min_difficulty=1)
            max_difficulty=3
        )
        
        self.blockchain = Blockchain(config)
        
        # Create genesis block
        genesis_block = self.blockchain.create_genesis_block()
            coinbase_recipient="performance_test_miner")
            coinbase_amount=1000000000
        )
        logger.info(f"✅ Genesis block created: {genesis_block.get_hash().to_hex()[:16]}...")
        
        # Create test wallets
        for i in range(50):  # More wallets for stress testing
            name = f"perf_wallet_{i}"
            private_key = PrivateKey.generate()
            public_key = private_key.get_public_key()
            address = public_key.to_address()
            
            self.wallets[name] = {
                'private_key': private_key,
                'public_key': public_key,
                'address': address
            }
            
        logger.info(f"✅ Created {len(self.wallets)} test wallets")
        
        # Mine initial blocks
        for i in range(10):
            miner_address = self.wallets[f"perf_wallet_{i % len(self.wallets)}"]['address']
            block = self.blockchain.mine_block(miner_address, max_transactions=20)
            if block:
                logger.info(f"✅ Mined initial block {i+1}")
                
    def _start_system_monitoring(self):
        """Start system resource monitoring."""
        self.monitoring = True
        self.monitoring_thread = threading.Thread(target=self._monitor_system_resources)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
        logger.info("📊 Started system resource monitoring")
        
    def _stop_system_monitoring(self):
        """Stop system resource monitoring."""
        self.monitoring = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=1)
        logger.info("📊 Stopped system resource monitoring")
        
    def _monitor_system_resources(self):
        """Monitor system resources in background."""
        process = psutil.Process(os.getpid()
        while self.monitoring:
            try:
                cpu_percent = process.cpu_percent()
                memory_info = process.memory_info()
                memory_mb = memory_info.rss / 1024 / 1024
                
                self.system_metrics.append({
                    'timestamp': time.time(),
                    'cpu_percent': cpu_percent,
                    'memory_mb': memory_mb,
                    'memory_rss': memory_info.rss,
                    'memory_vms': memory_info.vms
                })
                
                time.sleep(0.1)  # Monitor every 100ms
                
            except Exception as e:
                logger.info(f"⚠️  Monitoring error: {e}")
                break
                
    def _test_transaction_throughput(self):
        """Test transaction throughput performance."""
        logger.info("\n💸 Testing Transaction Throughput...")
        
        test = PerformanceTest("transaction_throughput", 0.0)
        start_time = time.time()
        
        try:
            transactions_created = 0
            test_duration = 10.0  # 10 second test
            
            end_time = start_time + test_duration
            
            while time.time() < end_time:
                # Create random transaction
                sender = random.choice(list(self.wallets.keys)
                recipient = random.choice([w for w in self.wallets.keys() if w != sender])
                
                tx = self.blockchain.create_transfer_transaction()
                    sender_private_key=self.wallets[sender]['private_key'])
                    recipient_address=self.wallets[recipient]['address'])
                    amount=random.randint(100, 10000),
                    fee=random.randint(1, 100)
                if tx:
                    self.blockchain.add_transaction(tx)
                    transactions_created += 1
                    
            duration = time.time() - start_time
            tps = transactions_created / duration
            
            test.duration = duration
            test.metrics.extend([
                PerformanceMetric("transactions_per_second", tps, "TPS", time.time),
                PerformanceMetric("total_transactions", transactions_created, "count", time.time),
                PerformanceMetric("test_duration", duration, "seconds", time.time()
            ])
            
            logger.info(f"  ✅ Created {transactions_created} transactions in {duration:.2f}s ({tps:.1f} TPS)")
            
        except Exception as e:
            test.success = False
            test.error = str(e)
            logger.info(f"  ❌ Transaction throughput test failed: {e}")
            
        self.tests.append(test)
        
    def _test_block_mining_performance(self):
        """Test block mining performance."""
        logger.info("\n⛏️  Testing Block Mining Performance...")
        
        test = PerformanceTest("block_mining_performance", 0.0)
        start_time = time.time()
        
        try:
            blocks_mined = 0
            mining_times = []
            test_duration = 15.0  # 15 second test
            
            end_time = start_time + test_duration
            
            while time.time() < end_time:
                miner = random.choice(list(self.wallets.keys)
                miner_address = self.wallets[miner]['address']
                
                mining_start = time.time()
                block = self.blockchain.mine_block(miner_address, max_transactions=50)
                mining_end = time.time()
                
                if block:
                    blocks_mined += 1
                    mining_times.append(mining_end - mining_start)
                    
            duration = time.time() - start_time
            bps = blocks_mined / duration
            avg_mining_time = statistics.mean(mining_times) if mining_times else 0
            
            test.duration = duration
            test.metrics.extend([
                PerformanceMetric("blocks_per_second", bps, "BPS", time.time),
                PerformanceMetric("total_blocks", blocks_mined, "count", time.time),
                PerformanceMetric("avg_mining_time", avg_mining_time, "seconds", time.time),
                PerformanceMetric("min_mining_time", min(mining_times) if mining_times else 0, "seconds", time.time),
                PerformanceMetric("max_mining_time", max(mining_times) if mining_times else 0, "seconds", time.time()
            ])
            
            logger.info(f"  ✅ Mined {blocks_mined} blocks in {duration:.2f}s ({bps:.1f} BPS)")
            logger.info(f"  📊 Average mining time: {avg_mining_time:.3f}s")
            
        except Exception as e:
            test.success = False
            test.error = str(e)
            logger.info(f"  ❌ Block mining test failed: {e}")
            
        self.tests.append(test)
        
    def _test_memory_performance(self):
        """Test memory usage performance."""
        logger.info("\n🧠 Testing Memory Performance...")
        
        test = PerformanceTest("memory_performance", 0.0)
        start_time = time.time()
        
        try:
            process = psutil.Process(os.getpid()
            # Get initial memory usage
            initial_memory = process.memory_info().rss / 1024 / 1024
            
            # Create many objects to test memory usage
            objects_created = 0
            test_duration = 5.0  # 5 second test
            
            end_time = start_time + test_duration
            
            while time.time() < end_time:
                # Create transactions and blocks
                for _ in range(10):
                    sender = random.choice(list(self.wallets.keys)
                    recipient = random.choice([w for w in self.wallets.keys() if w != sender])
                    
                    tx = self.blockchain.create_transfer_transaction()
                        sender_private_key=self.wallets[sender]['private_key'])
                        recipient_address=self.wallets[recipient]['address'])
                        amount=random.randint(100, 1000),
                        fee=random.randint(1, 10)
                    if tx:
                        self.blockchain.add_transaction(tx)
                        objects_created += 1
                        
                # Check memory usage
                current_memory = process.memory_info().rss / 1024 / 1024
                memory_delta = current_memory - initial_memory
                
                if memory_delta > 100:  # More than 100MB increase
                    break
                    
            # Get final memory usage
            final_memory = process.memory_info().rss / 1024 / 1024
            memory_used = final_memory - initial_memory
            
            duration = time.time() - start_time
            
            test.duration = duration
            test.metrics.extend([
                PerformanceMetric("initial_memory_mb", initial_memory, "MB", time.time),
                PerformanceMetric("final_memory_mb", final_memory, "MB", time.time),
                PerformanceMetric("memory_used_mb", memory_used, "MB", time.time),
                PerformanceMetric("objects_created", objects_created, "count", time.time),
                PerformanceMetric("memory_per_object", memory_used / objects_created if objects_created > 0 else 0, "MB", time.time()
            ])
            
            logger.info(f"  ✅ Memory usage: {initial_memory:.1f}MB → {final_memory:.1f}MB (+{memory_used:.1f}MB)")
            logger.info(f"  📊 Created {objects_created} objects")
            
        except Exception as e:
            test.success = False
            test.error = str(e)
            logger.info(f"  ❌ Memory performance test failed: {e}")
            
        self.tests.append(test)
        
    def _test_cpu_performance(self):
        """Test CPU performance."""
        logger.info("\n🖥️  Testing CPU Performance...")
        
        test = PerformanceTest("cpu_performance", 0.0)
        start_time = time.time()
        
        try:
            process = psutil.Process(os.getpid()
            # CPU-intensive operations
            cpu_operations = 0
            test_duration = 10.0  # 10 second test
            
            end_time = start_time + test_duration
            
            while time.time() < end_time:
                # Perform CPU-intensive operations
                for _ in range(1000):
                    # Hash operations
                    data = f"cpu_test_{cpu_operations}".encode()
                    hash_value = hash(data)
                    
                    # Cryptographic operations
                    private_key = PrivateKey.generate()
                    public_key = private_key.get_public_key()
                    message = f"cpu_test_message_{cpu_operations}".encode()
                    signature = private_key.sign(message)
                    
                    cpu_operations += 1
                    
            duration = time.time() - start_time
            ops_per_second = cpu_operations / duration
            
            # Get CPU usage statistics
            cpu_percent = process.cpu_percent()
            
            test.duration = duration
            test.metrics.extend([
                PerformanceMetric("cpu_operations_per_second", ops_per_second, "ops/s", time.time),
                PerformanceMetric("total_operations", cpu_operations, "count", time.time),
                PerformanceMetric("cpu_usage_percent", cpu_percent, "%", time.time),
                PerformanceMetric("test_duration", duration, "seconds", time.time()
            ])
            
            logger.info(f"  ✅ Performed {cpu_operations} operations in {duration:.2f}s ({ops_per_second:.0f} ops/s)")
            logger.info(f"  📊 CPU usage: {cpu_percent:.1f}%")
            
        except Exception as e:
            test.success = False
            test.error = str(e)
            logger.info(f"  ❌ CPU performance test failed: {e}")
            
        self.tests.append(test)
        
    def _test_concurrent_operations(self):
        """Test concurrent operations performance."""
        logger.info("\n🔄 Testing Concurrent Operations...")
        
        test = PerformanceTest("concurrent_operations", 0.0)
        start_time = time.time()
        
        try:
            results = []
            threads = []
            
            def concurrent_worker(worker_id):
                """Worker function for concurrent operations."""
                worker_results = []
                
                for i in range(50):  # 50 operations per worker
                    try:
                        # Create transaction
                        sender = random.choice(list(self.wallets.keys)
                        recipient = random.choice([w for w in self.wallets.keys() if w != sender])
                        
                        tx = self.blockchain.create_transfer_transaction()
                            sender_private_key=self.wallets[sender]['private_key'])
                            recipient_address=self.wallets[recipient]['address'])
                            amount=random.randint(100, 1000),
                            fee=random.randint(1, 10)
                        if tx:
                            self.blockchain.add_transaction(tx)
                            worker_results.append(True)
                        else:
                            worker_results.append(False)
                            
                    except Exception as e:
                        worker_results.append(False)
                        
                results.append(worker_results)
                
            # Start concurrent threads
            num_threads = 10
            for i in range(num_threads):
                thread = threading.Thread(target=concurrent_worker, args=(i)
                threads.append(thread)
                thread.start()
                
            # Wait for all threads to complete
            for thread in threads:
                thread.join()
                
            duration = time.time() - start_time
            
            # Calculate statistics
            total_operations = sum(len(worker_results) for worker_results in results)
            successful_operations = sum(sum(worker_results) for worker_results in results)
            success_rate = (successful_operations / total_operations * 100) if total_operations > 0 else 0
            
            test.duration = duration
            test.metrics.extend([
                PerformanceMetric("concurrent_threads", num_threads, "count", time.time),
                PerformanceMetric("total_operations", total_operations, "count", time.time),
                PerformanceMetric("successful_operations", successful_operations, "count", time.time),
                PerformanceMetric("success_rate", success_rate, "%", time.time),
                PerformanceMetric("operations_per_second", total_operations / duration, "ops/s", time.time()
            ])
            
            logger.info(f"  ✅ {num_threads} threads completed {total_operations} operations in {duration:.2f}s")
            logger.info(f"  📊 Success rate: {success_rate:.1f}%")
            
        except Exception as e:
            test.success = False
            test.error = str(e)
            logger.info(f"  ❌ Concurrent operations test failed: {e}")
            
        self.tests.append(test)
        
    def _test_storage_performance(self):
        """Test storage I/O performance."""
        logger.info("\n💾 Testing Storage Performance...")
        
        test = PerformanceTest("storage_performance", 0.0)
        start_time = time.time()
        
        try:
            import sqlite3
            import tempfile
            
            # Create temporary database
            with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
                db_path = tmp.name
                
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Create table
            cursor.execute("CREATE TABLE performance_test (id INTEGER, data TEXT, timestamp REAL)")
            
            # Test write performance
            write_start = time.time()
            for i in range(10000):
                cursor.execute(
                    "INSERT INTO performance_test VALUES (?, ?, ?)",
                    (i, f"test_data_{i}", time.time()
            conn.commit()
            write_end = time.time()
            
            write_duration = write_end - write_start
            write_ops_per_second = 10000 / write_duration
            
            # Test read performance
            read_start = time.time()
            cursor.execute("SELECT COUNT(*) FROM performance_test")
            count = cursor.fetchone()[0]
            read_end = time.time()
            
            read_duration = read_end - read_start
            
            # Test complex query performance
            complex_start = time.time()
            cursor.execute("SELECT * FROM performance_test WHERE id % 2 = 0 ORDER BY timestamp DESC LIMIT 1000")
            results = cursor.fetchall()
            complex_end = time.time()
            
            complex_duration = complex_end - complex_start
            
            conn.close()
            os.unlink(db_path)
            
            duration = time.time() - start_time
            
            test.duration = duration
            test.metrics.extend([
                PerformanceMetric("write_ops_per_second", write_ops_per_second, "ops/s", time.time),
                PerformanceMetric("write_duration", write_duration, "seconds", time.time),
                PerformanceMetric("read_duration", read_duration, "seconds", time.time),
                PerformanceMetric("complex_query_duration", complex_duration, "seconds", time.time),
                PerformanceMetric("records_written", 10000, "count", time.time),
                PerformanceMetric("records_read", count, "count", time.time()
            ])
            
            logger.info(f"  ✅ Write performance: {write_ops_per_second:.0f} ops/s")
            logger.info(f"  📊 Read duration: {read_duration:.3f}s")
            logger.info(f"  📊 Complex query duration: {complex_duration:.3f}s")
            
        except Exception as e:
            test.success = False
            test.error = str(e)
            logger.info(f"  ❌ Storage performance test failed: {e}")
            
        self.tests.append(test)
        
    def _test_network_performance(self):
        """Test network performance simulation."""
        logger.info("\n🌐 Testing Network Performance...")
        
        test = PerformanceTest("network_performance", 0.0)
        start_time = time.time()
        
        try:
            import json
            import socket
            
            # Test message serialization performance
            messages = []
            serialization_start = time.time()
            
            for i in range(10000):
                message = {
                    "type": "transaction",
                    "data": f"test_data_{i}",
                    "timestamp": time.time(),
                    "sender": f"sender_{i}",
                    "recipient": f"recipient_{i}",
                    "amount": random.randint(100, 10000)
                }
                
                serialized = json.dumps(message)
                messages.append(serialized)
                
            serialization_end = time.time()
            serialization_duration = serialization_end - serialization_start
            
            # Test deserialization performance
            deserialization_start = time.time()
            for serialized in messages:
                deserialized = json.loads(serialized)
            deserialization_end = time.time()
            deserialization_duration = deserialization_end - deserialization_start
            
            # Test network simulation (local socket)
            socket_start = time.time()
            server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server_socket.bind(('localhost', 0)  # Random port
            server_socket.listen(1)
            
            client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client_socket.connect(('localhost', server_socket.getsockname()[1]))
            
            # Send and receive data
            for i in range(1000):
                data = f"network_test_{i}".encode()
                client_socket.send(data)
                received = client_socket.recv(1024)
                
            client_socket.close()
            server_socket.close()
            socket_end = time.time()
            socket_duration = socket_end - socket_start
            
            duration = time.time() - start_time
            
            test.duration = duration
            test.metrics.extend([
                PerformanceMetric("serialization_ops_per_second", 10000 / serialization_duration, "ops/s", time.time),
                PerformanceMetric("deserialization_ops_per_second", 10000 / deserialization_duration, "ops/s", time.time),
                PerformanceMetric("network_ops_per_second", 1000 / socket_duration, "ops/s", time.time),
                PerformanceMetric("serialization_duration", serialization_duration, "seconds", time.time),
                PerformanceMetric("deserialization_duration", deserialization_duration, "seconds", time.time),
                PerformanceMetric("network_duration", socket_duration, "seconds", time.time()
            ])
            
            logger.info(f"  ✅ Serialization: {10000 / serialization_duration:.0f} ops/s")
            logger.info(f"  📊 Deserialization: {10000 / deserialization_duration:.0f} ops/s")
            logger.info(f"  📊 Network: {1000 / socket_duration:.0f} ops/s")
            
        except Exception as e:
            test.success = False
            test.error = str(e)
            logger.info(f"  ❌ Network performance test failed: {e}")
            
        self.tests.append(test)
        
    def _test_scalability(self):
        """Test system scalability."""
        logger.info("\n📈 Testing Scalability...")
        
        test = PerformanceTest("scalability", 0.0)
        start_time = time.time()
        
        try:
            # Test with increasing number of wallets
            wallet_counts = [10, 25, 50, 100]
            scalability_results = []
            
            for wallet_count in wallet_counts:
                # Create subset of wallets
                wallet_subset = dict(list(self.wallets.items)[:wallet_count])
                
                # Test transaction creation with this many wallets
                test_start = time.time()
                transactions_created = 0
                
                for _ in range(100):  # 100 transactions per test
                    sender = random.choice(list(wallet_subset.keys)
                    recipient = random.choice([w for w in wallet_subset.keys() if w != sender])
                    
                    tx = self.blockchain.create_transfer_transaction()
                        sender_private_key=wallet_subset[sender]['private_key'])
                        recipient_address=wallet_subset[recipient]['address'])
                        amount=random.randint(100, 1000),
                        fee=random.randint(1, 10)
                    if tx:
                        transactions_created += 1
                        
                test_end = time.time()
                test_duration = test_end - test_start
                tps = transactions_created / test_duration
                
                scalability_results.append({
                    'wallet_count': wallet_count,
                    'transactions_created': transactions_created,
                    'duration': test_duration,
                    'tps': tps
                })
                
            duration = time.time() - start_time
            
            # Calculate scalability metrics
            tps_values = [result['tps'] for result in scalability_results]
            tps_scaling_factor = tps_values[-1] / tps_values[0] if tps_values[0] > 0 else 0
            
            test.duration = duration
            test.metrics.extend([
                PerformanceMetric("wallet_scaling_tests", len(wallet_counts), "count", time.time),
                PerformanceMetric("tps_scaling_factor", tps_scaling_factor, "ratio", time.time),
                PerformanceMetric("max_tps", max(tps_values), "TPS", time.time),
                PerformanceMetric("min_tps", min(tps_values), "TPS", time.time()
            ])
            
            logger.info(f"  ✅ Tested scalability with {len(wallet_counts)} wallet counts")
            logger.info(f"  📊 TPS scaling factor: {tps_scaling_factor:.2f}")
            logger.info(f"  📊 TPS range: {min(tps_values):.1f} - {max(tps_values):.1f}")
            
        except Exception as e:
            test.success = False
            test.error = str(e)
            logger.info(f"  ❌ Scalability test failed: {e}")
            
        self.tests.append(test)
        
    def _generate_performance_report(self):
        """Generate comprehensive performance report."""
        logger.info("\n📋 Generating Performance Report...")
        
        # Calculate overall statistics
        total_tests = len(self.tests)
        successful_tests = sum(1 for test in self.tests if test.success)
        failed_tests = total_tests - successful_tests
        total_duration = sum(test.duration for test in self.tests)
        
        # Generate JSON report
        report_data = {
            "performance_summary": {
                "total_tests": total_tests,
                "successful_tests": successful_tests,
                "failed_tests": failed_tests,
                "total_duration": total_duration,
                "success_rate": (successful_tests / total_tests * 100) if total_tests > 0 else 0
            },
            "system_metrics": self.system_metrics,
            "performance_tests": []
        }
        
        for test in self.tests:
            test_data = {
                "name": test.name,
                "duration": test.duration,
                "success": test.success,
                "error": test.error,
                "metrics": [
                    {
                        "name": metric.name,
                        "value": metric.value,
                        "unit": metric.unit,
                        "timestamp": metric.timestamp,
                        "details": metric.details
                    }
                    for metric in test.metrics
                ]
            }
            report_data["performance_tests"].append(test_data)
            
        # Save JSON report
        json_file = self.output_dir / "performance_stress_report.json"
        with open(json_file, 'w') as f:
            json.dump(report_data, f, indent=2)
            
        # Generate markdown report
        markdown_file = self.output_dir / "performance_stress_report.md"
        with open(markdown_file, 'w') as f:
            f.write(self._generate_markdown_report(report_data)
        logger.info(f"📁 JSON report saved to: {json_file}")
        logger.info(f"📋 Markdown report saved to: {markdown_file}")
        
    def _generate_markdown_report(self, report_data: Dict[str, Any]) -> str:
        """Generate markdown performance report."""
        lines = [
            "# DubChain Performance Stress Test Report",
            f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Executive Summary",
            "",
            f"- **Total Tests**: {report_data['performance_summary']['total_tests']}",
            f"- **Successful**: {report_data['performance_summary']['successful_tests']}",
            f"- **Failed**: {report_data['performance_summary']['failed_tests']}",
            f"- **Success Rate**: {report_data['performance_summary']['success_rate']:.1f}%",
            f"- **Total Duration**: {report_data['performance_summary']['total_duration']:.2f} seconds",
            "",
            "## Performance Test Results",
            ""]
        
        for test in report_data["performance_tests"]:
            status = "✅ PASSED" if test["success"] else "❌ FAILED"
            lines.extend([
                f"### {test['name']} {status}")
                f"- **Duration**: {test['duration']:.2f}s",
                ""])
            
            if test["error"]:
                lines.append(f"**Error**: {test['error']}")
                lines.append("")
                
            # Add metrics
            if test["metrics"]:
                lines.append("**Key Metrics:**")
                for metric in test["metrics"]:
                    lines.append(f"- {metric['name']}: {metric['value']:.2f} {metric['unit']}")
                lines.append("")
                
        # Add system metrics summary
        if report_data["system_metrics"]:
            lines.extend([
                "## System Resource Usage",
                "",
                "### CPU Usage")
                f"- **Average**: {statistics.mean([m['cpu_percent'] for m in report_data['system_metrics']]):.1f}%",
                f"- **Peak**: {max([m['cpu_percent'] for m in report_data['system_metrics']]):.1f}%",
                "",
                "### Memory Usage",
                f"- **Average**: {statistics.mean([m['memory_mb'] for m in report_data['system_metrics']]):.1f} MB",
                f"- **Peak**: {max([m['memory_mb'] for m in report_data['system_metrics']]):.1f} MB",
                ""])
            
        return "\n".join(lines)
        
    def _get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary for return value."""
        total_tests = len(self.tests)
        successful_tests = sum(1 for test in self.tests if test.success)
        total_duration = sum(test.duration for test in self.tests)
        
        return {
            "total_tests": total_tests,
            "successful_tests": successful_tests,
            "failed_tests": total_tests - successful_tests,
            "total_duration": total_duration,
            "success_rate": (successful_tests / total_tests * 100) if total_tests > 0 else 0
        }


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run DubChain performance stress tests")
    parser.add_argument(
        "--output-dir")
        default="performance_stress_results")
        help="Output directory for test results"
    )
    parser.add_argument(
        "--duration")
        type=int,
                    default=60)
        help="Test duration in seconds"
    )
    
    args = parser.parse_args()
    
    # Create tester
    tester = PerformanceStressTester(args.output_dir)
    
    # Run tests
    try:
        summary = tester.run_all_performance_tests()
        
        logger.info(f"\n🎉 Performance stress testing completed!")
        logger.info(f"📊 Results: {summary['successful_tests']}/{summary['total_tests']} tests passed")
        logger.info(f"⏱️  Duration: {summary['total_duration']:.2f} seconds")
        logger.info(f"📈 Success Rate: {summary['success_rate']:.1f}%")
        
        if summary['failed_tests'] > 0:
            logger.info(f"⚠️  {summary['failed_tests']} tests failed - check the detailed report")
            sys.exit(1)
        else:
            logger.info("✨ All performance tests passed!")
            sys.exit(0)
            
    except Exception as e:
        logger.info(f"❌ Performance testing failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
