# Testing Guide

This document explains the testing framework and strategies used in DubChain.

## Testing Framework

### Test Categories
- **Unit Tests**: Individual component testing
- **Integration Tests**: Component interaction testing
- **Property-Based Tests**: Edge case discovery
- **Performance Tests**: Benchmarking and optimization
- **Security Tests**: Security and robustness testing

### Test Structure
```
tests/
├── unit/           # Unit tests
├── integration/    # Integration tests
├── property/       # Property-based tests
├── performance/    # Performance tests
└── fixtures/       # Test fixtures
```

## Running Tests

### All Tests
```bash
pytest
```

### Specific Test Categories
```bash
pytest -m unit          # Unit tests
pytest -m integration   # Integration tests
pytest -m performance   # Performance tests
```

### With Coverage
```bash
pytest --cov=dubchain --cov-report=html
```

## Writing Tests

### Unit Test Example
```python
def test_block_creation():
    block = Block(
        index=1,
        timestamp=time.time(),
        transactions=[],
        previous_hash="0"
    )
    assert block.index == 1
    assert block.hash is not None
```

### Integration Test Example
```python
def test_blockchain_operations():
    blockchain = Blockchain()
    wallet = Wallet.generate()
    
    tx = wallet.create_transaction("recipient", 100)
    blockchain.add_transaction(tx)
    block = blockchain.mine_block()
    
    assert len(blockchain.chain) == 2
    assert block.transactions[0] == tx
```

## Further Reading

- [Development Setup](../development/README.md)
- [Contributing Guide](../contributing/README.md)
