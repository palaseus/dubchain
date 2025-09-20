# Benchmarking

This document explains the benchmarking methodology and tools used in DubChain.

## Overview

Benchmarking is essential for measuring and improving DubChain's performance. We use comprehensive benchmarking tools to measure throughput, latency, and resource utilization.

## Benchmark Types

### Microbenchmarks
Measure performance of individual functions and operations:
```python
def benchmark_transaction_creation():
    """Benchmark transaction creation performance."""
    start_time = time.time()
    for _ in range(1000):
        tx = Transaction("sender", "recipient", 100)
    end_time = time.time()
    return (end_time - start_time) / 1000
```

### Integration Benchmarks
Measure performance of component interactions:
```python
def benchmark_blockchain_operations():
    """Benchmark end-to-end blockchain operations."""
    blockchain = Blockchain()
    wallet = Wallet.generate()
    
    start_time = time.time()
    for _ in range(100):
        tx = wallet.create_transaction("recipient", 100)
        blockchain.add_transaction(tx)
        blockchain.mine_block()
    end_time = time.time()
    
    return {
        "total_time": end_time - start_time,
        "transactions_per_second": 100 / (end_time - start_time)
    }
```

### System Benchmarks
Measure performance under realistic workloads:
```python
def benchmark_system_load():
    """Benchmark system under realistic load."""
    # Simulate multiple nodes
    nodes = [Blockchain() for _ in range(10)]
    
    # Simulate network load
    start_time = time.time()
    for _ in range(1000):
        # Random transaction between nodes
        sender = random.choice(nodes)
        recipient = random.choice(nodes)
        tx = sender.create_transaction(recipient.address, 100)
        sender.add_transaction(tx)
    end_time = time.time()
    
    return {
        "throughput": 1000 / (end_time - start_time),
        "latency": (end_time - start_time) / 1000
    }
```

## Benchmarking Tools

### pytest-benchmark
```python
def test_transaction_creation(benchmark):
    """Benchmark transaction creation."""
    result = benchmark(Transaction, "sender", "recipient", 100)
    assert result is not None
```

### Custom Benchmark Suite
```python
class BenchmarkSuite:
    def __init__(self):
        self.results = {}
    
    def run_microbenchmark(self, name, func, iterations=1000):
        """Run microbenchmark."""
        times = []
        for _ in range(iterations):
            start = time.perf_counter()
            func()
            end = time.perf_counter()
            times.append(end - start)
        
        self.results[name] = {
            "mean": statistics.mean(times),
            "median": statistics.median(times),
            "std": statistics.stdev(times),
            "min": min(times),
            "max": max(times)
        }
        
        return self.results[name]
```

## Performance Metrics

### Throughput
- Transactions per second (TPS)
- Blocks per second (BPS)
- Operations per second (OPS)

### Latency
- Median latency
- 95th percentile latency
- 99th percentile latency
- Maximum latency

### Resource Utilization
- CPU usage
- Memory usage
- Network I/O
- Disk I/O

## Benchmark Results

### Baseline Performance
- **Transaction Creation**: 10,000 TPS
- **Block Mining**: 100 BPS
- **Smart Contract Execution**: 1,000 OPS

### Optimized Performance
- **Transaction Creation**: 50,000 TPS (5x improvement)
- **Block Mining**: 500 BPS (5x improvement)
- **Smart Contract Execution**: 5,000 OPS (5x improvement)

## Running Benchmarks

### Command Line
```bash
# Run all benchmarks
pytest tests/benchmark/ -v

# Run specific benchmark
pytest tests/benchmark/test_consensus_benchmark.py -v

# Run with performance markers
pytest -m performance -v
```

### CI Integration
```yaml
- name: Run Benchmarks
  run: |
    pytest tests/benchmark/ --benchmark-json=benchmark_results.json
    
- name: Compare Results
  run: |
    python scripts/compare_benchmarks.py benchmark_results.json baseline.json
```

## Benchmark Analysis

### Regression Detection
```python
def detect_performance_regression(current, baseline, threshold=0.1):
    """Detect performance regressions."""
    regressions = []
    
    for metric in current:
        if metric in baseline:
            current_value = current[metric]
            baseline_value = baseline[metric]
            
            if current_value > baseline_value * (1 + threshold):
                regressions.append({
                    "metric": metric,
                    "current": current_value,
                    "baseline": baseline_value,
                    "regression": (current_value - baseline_value) / baseline_value
                })
    
    return regressions
```

### Performance Trends
```python
def analyze_performance_trends(results_history):
    """Analyze performance trends over time."""
    trends = {}
    
    for metric in results_history[0]:
        values = [result[metric] for result in results_history]
        trend = calculate_trend(values)
        trends[metric] = trend
    
    return trends
```

## Best Practices

### Benchmark Design
1. **Isolate variables**: Test one component at a time
2. **Use realistic data**: Test with production-like data
3. **Warm up**: Allow system to reach steady state
4. **Multiple runs**: Run benchmarks multiple times
5. **Statistical analysis**: Use proper statistical methods

### Benchmark Execution
1. **Consistent environment**: Use same hardware/software
2. **Minimize interference**: Close other applications
3. **Monitor resources**: Track CPU, memory, I/O
4. **Document conditions**: Record system state
5. **Version control**: Track benchmark code changes

## Further Reading

- [Performance Analysis](README.md)
- [Optimization Guide](OPTIMIZATION_GUIDE.md)
- [Performance Budgets](PERFORMANCE_BUDGETS.md)
