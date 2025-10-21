# ML Module Tests

This directory contains comprehensive unit and integration tests for all ML components.

## Test Files

- `test_ml_modules.py` - Core functionality tests for all ML modules
- `test_ml_performance.py` - Performance benchmarking and stress tests
- `test_config.py` - Test configuration and utilities

## Running Tests

### Run all ML tests:
```bash
python -m pytest tests/ml/ -v
```

### Run specific test categories:
```bash
# Core functionality tests
python -m pytest tests/ml/test_ml_modules.py -v

# Performance tests
python -m pytest tests/ml/test_ml_performance.py -v

# Infrastructure tests
python -m pytest tests/ml/ -k "infrastructure" -v

# Feature engineering tests
python -m pytest tests/ml/ -k "feature" -v

# Network topology tests
python -m pytest tests/ml/ -k "topology" -v

# Routing optimization tests
python -m pytest tests/ml/ -k "routing" -v

# Anomaly detection tests
python -m pytest tests/ml/ -k "anomaly" -v

# Bayesian optimization tests
python -m pytest tests/ml/ -k "optimization" -v

# Performance tests
python -m pytest tests/ml/ -k "performance" -v

# Stress tests
python -m pytest tests/ml/ -k "stress" -v
```

### Run with specific configurations:
```bash
# Enable stress tests
ENABLE_STRESS_TESTS=true python -m pytest tests/ml/ -v

# Disable specific test types
ENABLE_PERFORMANCE_TESTS=false python -m pytest tests/ml/ -v

# Set test data parameters
TEST_DATA_SIZE=5000 python -m pytest tests/ml/ -v

# Set performance thresholds
FEATURE_EXTRACTION_TIMEOUT=60.0 python -m pytest tests/ml/ -v

# Set memory limits
MAX_MEMORY_INCREASE_MB=2000 python -m pytest tests/ml/ -v
```

## Test Categories

### 1. Core Functionality Tests
- ML infrastructure tests
- Feature engineering tests
- Network topology optimization tests
- Reinforcement learning tests
- Anomaly detection tests
- Bayesian optimization tests
- Model performance tests
- Integration tests

### 2. Performance Tests
- Feature extraction performance
- Topology optimization performance
- Routing optimization performance
- Anomaly detection performance
- Parameter optimization performance
- Memory usage tests
- Throughput tests
- Latency tests

### 3. Stress Tests
- High volume feature extraction
- Large network topology optimization
- Continuous anomaly detection
- Concurrent ML operations
- Memory stress tests
- Long-running stability tests

## Test Configuration

Tests can be configured using environment variables:

- `ENABLE_INFRASTRUCTURE_TESTS` - Enable/disable infrastructure tests (default: true)
- `ENABLE_FEATURE_TESTS` - Enable/disable feature tests (default: true)
- `ENABLE_TOPOLOGY_TESTS` - Enable/disable topology tests (default: true)
- `ENABLE_ROUTING_TESTS` - Enable/disable routing tests (default: true)
- `ENABLE_ANOMALY_TESTS` - Enable/disable anomaly tests (default: true)
- `ENABLE_OPTIMIZATION_TESTS` - Enable/disable optimization tests (default: true)
- `ENABLE_PERFORMANCE_TESTS` - Enable/disable performance tests (default: true)
- `ENABLE_STRESS_TESTS` - Enable/disable stress tests (default: false)

### Test Data Parameters
- `TEST_DATA_SIZE` - Size of test datasets (default: 1000)
- `TEST_FEATURE_DIM` - Number of features in test data (default: 10)
- `TEST_NETWORK_SIZE` - Size of test networks (default: 50)

### Performance Thresholds
- `FEATURE_EXTRACTION_TIMEOUT` - Feature extraction timeout (default: 30.0s)
- `TOPOLOGY_OPTIMIZATION_TIMEOUT` - Topology optimization timeout (default: 60.0s)
- `ROUTING_OPTIMIZATION_TIMEOUT` - Routing optimization timeout (default: 30.0s)
- `ANOMALY_DETECTION_TIMEOUT` - Anomaly detection timeout (default: 120.0s)
- `PARAMETER_OPTIMIZATION_TIMEOUT` - Parameter optimization timeout (default: 180.0s)

### Memory Limits
- `MAX_MEMORY_INCREASE_MB` - Maximum memory increase in MB (default: 1000)

## Expected Results

### Core Functionality Tests
- All ML modules should initialize correctly
- Feature extraction should work properly
- Topology optimization should improve network performance
- Routing optimization should find optimal paths
- Anomaly detection should identify suspicious patterns
- Parameter optimization should find optimal parameters

### Performance Tests
- Feature extraction should complete within time limits
- Topology optimization should scale with network size
- Routing optimization should find paths quickly
- Anomaly detection should process data efficiently
- Parameter optimization should converge to optimal values

### Stress Tests
- High volume operations should complete successfully
- Large networks should be handled efficiently
- Continuous operations should maintain stability
- Concurrent operations should not interfere
- Memory usage should remain within limits

## Troubleshooting

### Common Issues

1. **Test Timeouts**
   - Increase timeout values
   - Check system resources
   - Verify test data size

2. **Memory Issues**
   - Increase memory limits
   - Reduce test data size
   - Check for memory leaks

3. **Performance Issues**
   - Check system resources
   - Verify ML libraries are installed
   - Consider using GPU acceleration

4. **Import Errors**
   - Check Python path
   - Verify ML dependencies are installed
   - Check module structure

### Debug Mode

Run tests with debug output:
```bash
python -m pytest tests/ml/ -v -s --tb=long
```

### Coverage Analysis

Generate test coverage report:
```bash
python -m pytest tests/ml/ --cov=src/dubchain/ml --cov-report=html
```

## Test Data

Test data is stored in the `test_data/` directory and includes:
- Sample transactions
- Test network topologies
- ML model data
- Performance benchmarks
- Stress test results

## Continuous Integration

Tests are designed to run in CI/CD pipelines with:
- Automated test execution
- Performance benchmarking
- Memory usage monitoring
- Coverage reporting
- Result archiving

## ML Dependencies

Tests require the following ML libraries:
- PyTorch
- Scikit-learn
- NumPy
- Pandas
- NetworkX (for graph operations)
- OpenTelemetry (for tracing)

Install dependencies:
```bash
pip install torch scikit-learn numpy pandas networkx opentelemetry-api
```
