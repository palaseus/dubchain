# Hardware Acceleration Tests

This directory contains comprehensive tests for the hardware acceleration modules.

## Test Files

- `test_hardware_acceleration.py` - Core functionality tests for all hardware accelerators
- `test_hardware_performance.py` - Performance benchmarking and stress tests
- `test_config.py` - Test configuration and utilities

## Running Tests

### Run all hardware tests:
```bash
python -m pytest tests/hardware/ -v
```

### Run specific test categories:
```bash
# Core functionality tests
python -m pytest tests/hardware/test_hardware_acceleration.py -v

# Performance tests
python -m pytest tests/hardware/test_hardware_performance.py -v

# CUDA-specific tests
python -m pytest tests/hardware/ -k "cuda" -v

# OpenCL-specific tests
python -m pytest tests/hardware/ -k "opencl" -v

# CPU-specific tests
python -m pytest tests/hardware/ -k "cpu" -v
```

### Run with specific configurations:
```bash
# Enable stress tests
ENABLE_STRESS_TESTS=true python -m pytest tests/hardware/ -v

# Disable CUDA tests
ENABLE_CUDA_TESTS=false python -m pytest tests/hardware/ -v

# Set benchmark iterations
BENCHMARK_ITERATIONS=10 python -m pytest tests/hardware/test_hardware_performance.py -v
```

## Test Categories

### 1. Core Functionality Tests
- Hardware detection and capability testing
- Accelerator initialization and configuration
- Basic operation execution
- Error handling and edge cases
- Integration between accelerators

### 2. Performance Benchmark Tests
- Matrix multiplication benchmarks
- Cryptographic operation benchmarks
- Memory allocation benchmarks
- Throughput and latency measurements
- Performance comparison across accelerators
- Scalability analysis

### 3. Stress Tests
- Continuous operation testing
- Large matrix handling
- Memory pressure testing
- Concurrent operation testing
- Long-running stability tests

## Test Configuration

Tests can be configured using environment variables:

- `ENABLE_CUDA_TESTS` - Enable/disable CUDA tests (default: true)
- `ENABLE_OPENCL_TESTS` - Enable/disable OpenCL tests (default: true)
- `ENABLE_CPU_TESTS` - Enable/disable CPU tests (default: true)
- `ENABLE_PERFORMANCE_TESTS` - Enable/disable performance tests (default: true)
- `ENABLE_STRESS_TESTS` - Enable/disable stress tests (default: false)
- `BENCHMARK_ITERATIONS` - Number of iterations for benchmarks (default: 5)
- `STRESS_TEST_DURATION` - Duration for stress tests in seconds (default: 30)

## Expected Results

### Performance Thresholds
- Matrix multiplication should complete within reasonable time limits
- CUDA should generally outperform CPU for large matrices
- OpenCL performance should be competitive with CUDA
- Memory allocation should be efficient and not leak

### Success Criteria
- All core functionality tests should pass
- Performance tests should meet minimum thresholds
- Stress tests should maintain stability
- Error handling should be robust

## Troubleshooting

### Common Issues

1. **CUDA Tests Failing**
   - Ensure CUDA toolkit is installed
   - Check GPU availability
   - Verify CUDA drivers are up to date

2. **OpenCL Tests Failing**
   - Ensure OpenCL runtime is installed
   - Check for compatible GPU/CPU
   - Verify OpenCL drivers

3. **Performance Tests Slow**
   - Check system resources
   - Ensure no other heavy processes running
   - Consider reducing benchmark iterations

4. **Memory Issues**
   - Check available system memory
   - Reduce matrix sizes for testing
   - Monitor memory usage during tests

### Debug Mode

Run tests with debug output:
```bash
python -m pytest tests/hardware/ -v -s --tb=long
```

### Coverage Analysis

Generate test coverage report:
```bash
python -m pytest tests/hardware/ --cov=src/dubchain/hardware --cov-report=html
```
