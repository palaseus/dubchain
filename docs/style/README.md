# Code Style Guide

This document outlines the coding standards and style guidelines for DubChain.

## Code Formatting

### Black
We use Black for automatic code formatting:
```bash
black src/ tests/
```

### Import Sorting
We use isort for import organization:
```bash
isort src/ tests/
```

## Style Guidelines

### Naming Conventions
- **Classes**: PascalCase (e.g., `BlockChain`)
- **Functions/Methods**: snake_case (e.g., `create_transaction`)
- **Variables**: snake_case (e.g., `block_hash`)
- **Constants**: UPPER_SNAKE_CASE (e.g., `MAX_BLOCK_SIZE`)

### Type Hints
Always use type hints for function parameters and return values:
```python
def create_transaction(sender: str, recipient: str, amount: int) -> Transaction:
    return Transaction(sender, recipient, amount)
```

### Documentation
Use docstrings for all public functions and classes:
```python
def mine_block(self, transactions: List[Transaction]) -> Block:
    """
    Mine a new block with the given transactions.
    
    Args:
        transactions: List of transactions to include in the block
        
    Returns:
        The newly mined block
        
    Raises:
        ValidationError: If transactions are invalid
    """
    pass
```

## Further Reading

- [Development Setup](../development/README.md)
- [Contributing Guide](../contributing/README.md)
