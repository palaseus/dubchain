# DubChain API Reference

This directory contains comprehensive API documentation for all DubChain modules and components.

## Core API Documentation

### Blockchain Core
- **[Block API](block.md)** - Block creation, validation, and management
- **[Transaction API](transaction.md)** - Transaction processing and validation
- **[Blockchain API](blockchain.md)** - Blockchain management and operations

### Consensus Mechanisms
- **[Consensus Engine API](consensus.md)** - Main consensus engine interface
- **[Proof of Stake API](pos.md)** - PoS consensus implementation
- **[Delegated Proof of Stake API](dpos.md)** - DPoS consensus implementation
- **[PBFT API](pbft.md)** - Practical Byzantine Fault Tolerance
- **[Hybrid Consensus API](hybrid.md)** - Hybrid consensus mechanisms

### Virtual Machine
- **[VM Execution Engine API](vm.md)** - Smart contract execution
- **[Opcodes API](opcodes.md)** - VM instruction set
- **[Gas Meter API](gas.md)** - Gas metering and optimization
- **[Contract API](contract.md)** - Smart contract management

### Cross-Chain Bridge
- **[Bridge Manager API](bridge.md)** - Bridge coordination and management
- **[Atomic Swap API](atomic-swap.md)** - Trustless cross-chain exchanges
- **[Cross-Chain Messaging API](messaging.md)** - Inter-chain communication
- **[Universal Assets API](assets.md)** - Universal asset management

### Sharding System
- **[Shard Manager API](sharding.md)** - Shard management and coordination
- **[Cross-Shard Communication API](cross-shard.md)** - Inter-shard messaging
- **[Shard Consensus API](shard-consensus.md)** - Shard-level consensus
- **[State Management API](state.md)** - Distributed state management

### Networking
- **[Peer Management API](peer.md)** - P2P peer management
- **[Gossip Protocol API](gossip.md)** - Message propagation
- **[Connection Manager API](connection.md)** - Network connection management
- **[Message Router API](router.md)** - Message routing and delivery

### Cryptography
- **[Digital Signatures API](signatures.md)** - ECDSA and other signature schemes
- **[Hash Functions API](hashing.md)** - Cryptographic hash functions
- **[Merkle Trees API](merkle.md)** - Merkle tree implementation
- **[Key Management API](keys.md)** - Key generation and management

### Performance Optimization
- **[Profiling API](profiling.md)** - Performance profiling and analysis
- **[Benchmarks API](benchmarks.md)** - Benchmarking and regression detection
- **[Optimizations API](optimizations.md)** - Optimization management and feature gates
- **[Monitoring API](monitoring.md)** - Real-time performance monitoring

## API Usage Examples

### Basic Blockchain Operations
```python
from dubchain import Blockchain, Wallet

# Create blockchain instance
blockchain = Blockchain()

# Create wallet
wallet = Wallet.generate()

# Create transaction
tx = wallet.create_transaction(
    recipient="recipient_address",
    amount=1000,
    fee=10
)

# Add to blockchain
blockchain.add_transaction(tx)
block = blockchain.mine_block()
```

### Smart Contract Deployment
```python
from dubchain.vm import SmartContract

# Deploy contract
contract = SmartContract.compile(contract_code)
tx = wallet.deploy_contract(contract)
blockchain.add_transaction(tx)
```

### Performance Optimization
```python
from dubchain.performance import OptimizationManager

# Enable optimizations
manager = OptimizationManager()
manager.enable_optimization("vm_bytecode_caching")
manager.enable_optimization("crypto_parallel_verification")
```

## API Design Principles

### Consistency
- All APIs follow consistent naming conventions
- Standardized parameter and return value formats
- Uniform error handling across all modules

### Type Safety
- Comprehensive type hints for all functions and methods
- Runtime type checking with typeguard
- Clear documentation of expected types

### Error Handling
- Structured exception hierarchy
- Detailed error messages with context
- Graceful degradation and fallback mechanisms

### Performance
- Optimized implementations with performance monitoring
- Configurable optimization levels
- Built-in benchmarking and profiling support

## Contributing to API Documentation

When adding new APIs:

1. **Document all public methods** with comprehensive docstrings
2. **Include type hints** for all parameters and return values
3. **Provide usage examples** for complex operations
4. **Update this index** with links to new documentation
5. **Test all examples** to ensure they work correctly

## API Versioning

DubChain follows semantic versioning for API changes:

- **Major version changes**: Breaking API changes
- **Minor version changes**: New features, backward compatible
- **Patch version changes**: Bug fixes, backward compatible

See [CHANGELOG.md](../../CHANGELOG.md) for detailed version history.
