# Contributing to DubChain

Thank you for your interest in contributing to DubChain! This guide will help you get started with contributing to our blockchain research and educational platform.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Process](#development-process)
- [Contributing Guidelines](#contributing-guidelines)
- [Code Style](#code-style)
- [Testing](#testing)
- [Documentation](#documentation)
- [Submitting Changes](#submitting-changes)
- [Review Process](#review-process)

## Code of Conduct

This project follows a code of conduct to ensure a welcoming environment for all contributors. Please be respectful, inclusive, and constructive in all interactions.

## Getting Started

### Prerequisites

- Python 3.10 or higher
- Git
- Basic understanding of blockchain concepts
- Familiarity with Python development

### Setting Up Development Environment

1. **Fork and clone the repository**:
   ```bash
   git clone https://github.com/your-username/dubchain.git
   cd dubchain
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install development dependencies**:
   ```bash
   pip install -e ".[dev]"
   pre-commit install
   ```

4. **Verify setup**:
   ```bash
   pytest tests/unit/test_core_block.py -v
   ```

## Development Process

### 1. Choose an Issue

- Browse [open issues](https://github.com/dubchain/dubchain/issues)
- Look for issues labeled `good first issue` for beginners
- Comment on the issue to indicate you're working on it

### 2. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/issue-number
```

### 3. Make Changes

- Write clean, well-documented code
- Follow the established code style
- Add tests for new functionality
- Update documentation as needed

### 4. Test Your Changes

```bash
# Run all tests
pytest

# Run specific test categories
pytest -m unit
pytest -m integration

# Run with coverage
pytest --cov=dubchain --cov-report=html
```

### 5. Submit a Pull Request

- Push your changes to your fork
- Create a pull request with a clear description
- Link to any related issues

## Contributing Guidelines

### Types of Contributions

We welcome various types of contributions:

#### Code Contributions
- **Bug Fixes**: Fix existing issues
- **Feature Development**: Add new features
- **Performance Improvements**: Optimize existing code
- **Refactoring**: Improve code structure and readability

#### Documentation Contributions
- **API Documentation**: Improve function and class documentation
- **Tutorials**: Create educational content
- **Examples**: Add practical usage examples
- **Research Papers**: Contribute academic research

#### Testing Contributions
- **Unit Tests**: Add tests for existing functionality
- **Integration Tests**: Test component interactions
- **Property-Based Tests**: Add Hypothesis-based tests
- **Performance Tests**: Add benchmarking tests

#### Research Contributions
- **Consensus Research**: Novel consensus mechanisms
- **Scalability Research**: Sharding and performance improvements
- **Security Research**: Security analysis and improvements
- **Interoperability Research**: Cross-chain protocol development

### Contribution Areas

#### High Priority
- **Bug Fixes**: Critical issues and security vulnerabilities
- **Documentation**: API documentation and tutorials
- **Testing**: Improving test coverage and quality
- **Performance**: Optimizing critical paths

#### Medium Priority
- **New Features**: Non-critical feature additions
- **Code Quality**: Refactoring and code improvements
- **Examples**: Additional usage examples
- **Research**: Academic research and papers

#### Low Priority
- **Cosmetic Changes**: UI improvements and formatting
- **Experimental Features**: Proof-of-concept implementations
- **Additional Tools**: Development and debugging tools

## Code Style

### Python Style Guide

We follow PEP 8 with some modifications:

- **Line Length**: 88 characters (Black default)
- **Import Order**: isort with Black profile
- **Type Hints**: Required for all public functions
- **Documentation**: Google-style docstrings

### Code Formatting

We use automated formatting tools:

```bash
# Format code
black src/ tests/
isort src/ tests/

# Type checking
mypy src/

# Run all checks
pre-commit run --all-files
```

### Naming Conventions

- **Classes**: PascalCase (e.g., `BlockValidator`)
- **Functions/Methods**: snake_case (e.g., `validate_block`)
- **Constants**: UPPER_SNAKE_CASE (e.g., `MAX_BLOCK_SIZE`)
- **Private Methods**: Leading underscore (e.g., `_internal_method`)

### Documentation Standards

#### Function Documentation

```python
def validate_transaction(self, transaction: Transaction) -> bool:
    """Validate a transaction.
    
    Args:
        transaction: The transaction to validate
        
    Returns:
        True if the transaction is valid, False otherwise
        
    Raises:
        ValidationError: If the transaction is invalid
        
    Example:
        >>> validator = TransactionValidator()
        >>> tx = Transaction(...)
        >>> validator.validate_transaction(tx)
        True
    """
```

#### Class Documentation

```python
class BlockValidator:
    """Validates blocks in the blockchain.
    
    This class provides comprehensive validation for blocks including
    transaction validation, signature verification, and consensus rules.
    
    Attributes:
        consensus_engine: The consensus engine to use for validation
        max_block_size: Maximum allowed block size in bytes
        
    Example:
        >>> validator = BlockValidator(consensus_engine)
        >>> block = Block(...)
        >>> validator.validate_block(block)
        True
    """
```

## Testing

### Test Categories

#### Unit Tests
- Test individual components in isolation
- Mock external dependencies
- Focus on specific functionality
- Fast execution

#### Integration Tests
- Test component interactions
- Use real dependencies where possible
- Test complete workflows
- Slower execution

#### Property-Based Tests
- Use Hypothesis for edge case discovery
- Test invariants and properties
- Generate random test data
- Comprehensive coverage

#### Performance Tests
- Benchmark critical operations
- Measure memory and CPU usage
- Test scalability
- Regression testing

### Writing Tests

#### Unit Test Example

```python
import pytest
from dubchain.core import Block, Transaction
from dubchain.core.block_validator import BlockValidator

class TestBlockValidator:
    """Test block validator functionality."""
    
    def test_validate_valid_block(self):
        """Test validation of a valid block."""
        validator = BlockValidator()
        block = Block.create_genesis_block()
        
        result = validator.validate_block(block)
        
        assert result is True
    
    def test_validate_invalid_block(self):
        """Test validation of an invalid block."""
        validator = BlockValidator()
        block = Block.create_genesis_block()
        block.hash = "invalid_hash"
        
        result = validator.validate_block(block)
        
        assert result is False
    
    @pytest.mark.parametrize("block_size", [100, 1000, 10000])
    def test_validate_block_size(self, block_size):
        """Test block size validation."""
        validator = BlockValidator()
        block = Block.create_test_block(size=block_size)
        
        result = validator.validate_block(block)
        
        assert result is True
```

#### Integration Test Example

```python
import pytest
from dubchain import Blockchain, Wallet

class TestBlockchainIntegration:
    """Test blockchain integration scenarios."""
    
    def test_complete_transaction_flow(self):
        """Test complete transaction flow from creation to confirmation."""
        blockchain = Blockchain()
        wallet = Wallet.generate()
        
        # Create transaction
        tx = wallet.create_transaction("recipient", 1000, 10)
        
        # Add to blockchain
        blockchain.add_transaction(tx)
        
        # Mine block
        block = blockchain.mine_block()
        
        # Verify transaction is confirmed
        assert tx.txid in [tx.txid for tx in block.transactions]
        assert blockchain.get_transaction(tx.txid) is not None
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific test categories
pytest -m unit
pytest -m integration
pytest -m crypto
pytest -m network

# Run with coverage
pytest --cov=dubchain --cov-report=html

# Run specific test file
pytest tests/unit/test_core_block.py

# Run specific test
pytest tests/unit/test_core_block.py::TestBlock::test_create_block
```

## Documentation

### Documentation Types

#### API Documentation
- Function and class docstrings
- Parameter and return value descriptions
- Usage examples
- Error conditions

#### Tutorial Documentation
- Step-by-step guides
- Complete examples
- Best practices
- Common pitfalls

#### Research Documentation
- Academic papers
- Design decisions
- Performance analysis
- Security considerations

### Writing Documentation

#### Docstring Example

```python
def create_transaction(self, recipient: str, amount: int, fee: int) -> Transaction:
    """Create a new transaction.
    
    Creates a new transaction that transfers the specified amount to the
    recipient address, paying the specified fee to miners.
    
    Args:
        recipient: The recipient's wallet address
        amount: The amount to transfer in the smallest unit
        fee: The transaction fee in the smallest unit
        
    Returns:
        A new Transaction object
        
    Raises:
        InsufficientFundsError: If the wallet has insufficient funds
        InvalidAddressError: If the recipient address is invalid
        
    Example:
        >>> wallet = Wallet.generate()
        >>> tx = wallet.create_transaction("0x1234...", 1000, 10)
        >>> print(tx.txid)
        0xabcd...
        
    Note:
        The transaction must be added to the blockchain and mined
        before it becomes confirmed.
    """
```

#### Tutorial Example

```markdown
# Creating Your First Smart Contract

This tutorial will guide you through creating and deploying a simple smart contract on DubChain.

## Prerequisites

- DubChain installed and running
- Basic understanding of smart contracts
- Python programming knowledge

## Step 1: Write the Contract

Create a simple storage contract:

```python
contract SimpleStorage {
    uint256 public storedData;
    
    function set(uint256 x) public {
        storedData = x;
    }
    
    function get() public view returns (uint256) {
        return storedData;
    }
}
```

## Step 2: Deploy the Contract

[Continue with deployment steps...]
```

## Submitting Changes

### Pull Request Process

1. **Create a Pull Request**:
   - Use a clear, descriptive title
   - Reference any related issues
   - Provide a detailed description

2. **Pull Request Template**:
   ```markdown
   ## Description
   Brief description of changes
   
   ## Type of Change
   - [ ] Bug fix
   - [ ] New feature
   - [ ] Documentation update
   - [ ] Performance improvement
   - [ ] Code refactoring
   
   ## Testing
   - [ ] Unit tests added/updated
   - [ ] Integration tests added/updated
   - [ ] All tests pass
   
   ## Documentation
   - [ ] API documentation updated
   - [ ] Tutorial documentation updated
   - [ ] README updated (if needed)
   
   ## Checklist
   - [ ] Code follows style guidelines
   - [ ] Self-review completed
   - [ ] No breaking changes (or documented)
   ```

3. **Link Issues**:
   - Use "Fixes #123" or "Closes #123" in the description
   - Reference related issues

### Commit Messages

Use clear, descriptive commit messages:

```bash
# Good commit messages
git commit -m "Fix transaction validation bug in block validator"
git commit -m "Add support for cross-chain atomic swaps"
git commit -m "Update API documentation for consensus module"

# Bad commit messages
git commit -m "fix bug"
git commit -m "update stuff"
git commit -m "WIP"
```

### Commit Message Format

```
type(scope): description

[optional body]

[optional footer]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes
- `refactor`: Code refactoring
- `test`: Test additions or changes
- `chore`: Maintenance tasks

Examples:
```
feat(consensus): add hybrid consensus mechanism
fix(blockchain): resolve chain reorganization bug
docs(api): update transaction validation documentation
test(vm): add property-based tests for gas metering
```

## Review Process

### Review Criteria

Pull requests are reviewed based on:

1. **Code Quality**:
   - Follows style guidelines
   - Well-documented
   - Proper error handling
   - Performance considerations

2. **Testing**:
   - Adequate test coverage
   - Tests are meaningful
   - All tests pass
   - Edge cases covered

3. **Documentation**:
   - API documentation updated
   - Usage examples provided
   - Breaking changes documented

4. **Functionality**:
   - Solves the intended problem
   - No regressions introduced
   - Backward compatibility maintained

### Review Process

1. **Automated Checks**:
   - All tests must pass
   - Code style checks must pass
   - Type checking must pass
   - Coverage requirements met

2. **Human Review**:
   - At least one maintainer review required
   - Code quality assessment
   - Architecture review (for major changes)
   - Security review (for security-related changes)

3. **Approval**:
   - All checks must pass
   - All requested changes addressed
   - Maintainer approval required

### Addressing Review Feedback

1. **Respond to Comments**:
   - Acknowledge feedback
   - Ask questions if unclear
   - Explain design decisions

2. **Make Changes**:
   - Address all requested changes
   - Add tests if requested
   - Update documentation if needed

3. **Re-request Review**:
   - Mark conversations as resolved
   - Request re-review when ready

## Getting Help

### Resources

- **Documentation**: [docs/README.md](../README.md)
- **Issues**: [GitHub Issues](https://github.com/dubchain/dubchain/issues)
- **Discussions**: [GitHub Discussions](https://github.com/dubchain/dubchain/discussions)
- **Email**: dev@dubchain.io

### Asking Questions

When asking questions:

1. **Search First**: Check existing issues and discussions
2. **Provide Context**: Include relevant code and error messages
3. **Be Specific**: Describe what you're trying to achieve
4. **Include Details**: Python version, OS, error logs

### Mentorship

We offer mentorship for new contributors:

- **Good First Issues**: Labeled issues suitable for beginners
- **Code Reviews**: Detailed feedback on pull requests
- **Documentation**: Comprehensive guides and tutorials
- **Community**: Helpful community members

## Recognition

Contributors are recognized through:

- **Contributors List**: Listed in project documentation
- **Release Notes**: Mentioned in release announcements
- **GitHub**: Contributor statistics and activity
- **Community**: Recognition in discussions and events

Thank you for contributing to DubChain! Your contributions help make blockchain technology more accessible and advance the field of distributed systems research.
