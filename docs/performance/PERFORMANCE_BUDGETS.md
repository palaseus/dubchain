# Performance Budgets

This document defines performance budgets and thresholds for DubChain components.

## Overview

Performance budgets establish clear performance targets and thresholds for critical system components. These budgets are enforced in CI/CD pipelines to prevent performance regressions.

## Performance Budgets

### Block Creation
- **Median Latency**: < 100ms
- **P95 Latency**: < 200ms
- **P99 Latency**: < 500ms
- **Throughput**: > 1000 tx/s

### Transaction Validation
- **Median Latency**: < 10ms
- **P95 Latency**: < 50ms
- **P99 Latency**: < 100ms
- **Throughput**: > 10000 tx/s

### Smart Contract Execution
- **Median Latency**: < 50ms
- **P95 Latency**: < 100ms
- **P99 Latency**: < 200ms
- **Gas Overhead**: < 5%

### Network Propagation
- **Block Propagation**: < 1s (95th percentile)
- **Transaction Propagation**: < 500ms (95th percentile)
- **Message Latency**: < 100ms (median)

### Storage Operations
- **Read Latency**: < 1ms (SSD), < 10ms (HDD)
- **Write Latency**: < 5ms (SSD), < 50ms (HDD)
- **Throughput**: > 10000 ops/s

### Memory Usage
- **Peak Memory**: < 2GB per node
- **Memory Growth**: < 10% per hour
- **GC Pause Time**: < 100ms

## Enforcement

### CI Integration
Performance budgets are enforced in the CI pipeline:
```yaml
- name: Performance Budget Check
  run: |
    python scripts/check_performance_budgets.py
    if [ $? -ne 0 ]; then
      echo "Performance budget exceeded"
      exit 1
    fi
```

### Monitoring
Real-time monitoring alerts when budgets are exceeded:
```python
def check_performance_budget(metric, value, budget):
    if value > budget:
        alert_manager.send_alert(
            f"Performance budget exceeded: {metric} = {value} > {budget}"
        )
        return False
    return True
```

## Further Reading

- [Performance Analysis](README.md)
- [Optimization Guide](OPTIMIZATION_GUIDE.md)
- [Benchmarking](BENCHMARKING.md)
