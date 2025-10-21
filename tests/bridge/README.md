# Bridge Integration Tests

This directory contains comprehensive integration tests for all blockchain bridge components.

## Test Files

- `test_bridge_integration.py` - Core integration tests for all bridge components
- `test_bridge_security.py` - Security and fraud detection tests
- `test_config.py` - Test configuration and utilities

## Running Tests

### Run all bridge tests:
```bash
python -m pytest tests/bridge/ -v
```

### Run specific test categories:
```bash
# Core integration tests
python -m pytest tests/bridge/test_bridge_integration.py -v

# Security tests
python -m pytest tests/bridge/test_bridge_security.py -v

# Ethereum-specific tests
python -m pytest tests/bridge/ -k "ethereum" -v

# Bitcoin-specific tests
python -m pytest tests/bridge/ -k "bitcoin" -v

# Polygon-specific tests
python -m pytest tests/bridge/ -k "polygon" -v

# BSC-specific tests
python -m pytest tests/bridge/ -k "bsc" -v

# Universal bridge tests
python -m pytest tests/bridge/ -k "universal" -v

# Validator network tests
python -m pytest tests/bridge/ -k "validator" -v

# Atomic swap tests
python -m pytest tests/bridge/ -k "atomic" -v

# Security tests
python -m pytest tests/bridge/ -k "security" -v

# Fraud detection tests
python -m pytest tests/bridge/ -k "fraud" -v

# Penetration tests
python -m pytest tests/bridge/ -k "penetration" -v
```

### Run with specific configurations:
```bash
# Enable penetration tests
ENABLE_PENETRATION_TESTS=true python -m pytest tests/bridge/ -v

# Disable specific chain tests
ENABLE_ETHEREUM_TESTS=false python -m pytest tests/bridge/ -v

# Set custom RPC URLs
ETHEREUM_RPC_URL=https://mainnet.infura.io/v3/YOUR_KEY python -m pytest tests/bridge/ -v

# Set custom timeouts
TRANSACTION_TIMEOUT=60 python -m pytest tests/bridge/ -v
```

## Test Categories

### 1. Core Integration Tests
- Ethereum bridge integration
- Bitcoin bridge integration
- Polygon bridge integration
- BSC bridge integration
- Universal bridge integration
- Bridge validator network integration
- Atomic swap integration
- Cross-chain transaction tests

### 2. Security Tests
- Fraud detection algorithm tests
- Security vulnerability tests
- Penetration testing scenarios
- Attack simulation tests
- Security audit tests

### 3. Performance Tests
- Transaction throughput tests
- Latency measurement tests
- Resource usage tests
- Scalability tests

## Test Configuration

Tests can be configured using environment variables:

- `ENABLE_ETHEREUM_TESTS` - Enable/disable Ethereum tests (default: true)
- `ENABLE_BITCOIN_TESTS` - Enable/disable Bitcoin tests (default: true)
- `ENABLE_POLYGON_TESTS` - Enable/disable Polygon tests (default: true)
- `ENABLE_BSC_TESTS` - Enable/disable BSC tests (default: true)
- `ENABLE_SECURITY_TESTS` - Enable/disable security tests (default: true)
- `ENABLE_PENETRATION_TESTS` - Enable/disable penetration tests (default: false)

### Network Configuration
- `ETHEREUM_RPC_URL` - Ethereum RPC endpoint
- `BITCOIN_RPC_HOST` - Bitcoin RPC host
- `BITCOIN_RPC_PORT` - Bitcoin RPC port
- `POLYGON_RPC_URL` - Polygon RPC endpoint
- `BSC_RPC_URL` - BSC RPC endpoint

### Timeout Configuration
- `TRANSACTION_TIMEOUT` - Transaction timeout in seconds (default: 30)
- `CONSENSUS_TIMEOUT` - Consensus timeout in seconds (default: 60)
- `SWAP_TIMEOUT` - Atomic swap timeout in seconds (default: 300)

## Expected Results

### Integration Tests
- All chain clients should initialize correctly
- Cross-chain transfers should complete successfully
- Validator consensus should work properly
- Atomic swaps should execute correctly

### Security Tests
- Fraud detection should identify suspicious patterns
- Security vulnerabilities should be detected
- Penetration tests should validate security measures
- Attack simulations should be properly handled

### Performance Tests
- Transactions should complete within reasonable time limits
- Throughput should meet minimum requirements
- Resource usage should be within acceptable limits

## Troubleshooting

### Common Issues

1. **RPC Connection Failures**
   - Check RPC endpoint URLs
   - Verify network connectivity
   - Ensure RPC credentials are correct

2. **Test Timeouts**
   - Increase timeout values
   - Check network latency
   - Verify system resources

3. **Mock Failures**
   - Check mock configurations
   - Verify test data
   - Ensure proper test setup

4. **Security Test Failures**
   - Check security configurations
   - Verify test patterns
   - Ensure proper test data

### Debug Mode

Run tests with debug output:
```bash
python -m pytest tests/bridge/ -v -s --tb=long
```

### Coverage Analysis

Generate test coverage report:
```bash
python -m pytest tests/bridge/ --cov=src/dubchain/bridge --cov-report=html
```

## Test Data

Test data is stored in the `test_data/` directory and includes:
- Sample transactions
- Test validators
- Swap proposals
- Network topologies
- Security patterns

## Continuous Integration

Tests are designed to run in CI/CD pipelines with:
- Automated test execution
- Coverage reporting
- Security scanning
- Performance benchmarking
- Result archiving
