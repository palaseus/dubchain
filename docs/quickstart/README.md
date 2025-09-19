# DubChain Quick Start Guide

This guide will help you get started with DubChain quickly. You'll learn the basic concepts and run your first blockchain operations.

## Prerequisites

- DubChain installed (see [Installation Guide](installation/README.md))
- Basic Python knowledge
- Understanding of blockchain concepts (helpful but not required)

## Basic Concepts

### Blockchain Fundamentals

A blockchain is a distributed ledger that maintains a continuously growing list of records (blocks) linked and secured using cryptography.

**Key Components:**
- **Blocks**: Containers for transactions and metadata
- **Transactions**: Operations that modify the blockchain state
- **Consensus**: Mechanism for agreeing on the valid state
- **Cryptography**: Security through digital signatures and hashing

### DubChain Architecture

DubChain implements:
- **UTXO Model**: Unspent Transaction Output model (like Bitcoin)
- **Multiple Consensus**: PoS, DPoS, PBFT, and Hybrid consensus
- **Smart Contracts**: Virtual machine for contract execution
- **Cross-Chain Bridge**: Interoperability with other blockchains
- **Sharding**: Horizontal scaling through network partitioning

## Your First Blockchain

### Step 1: Create a Blockchain

```python
from dubchain import Blockchain

# Create a new blockchain
blockchain = Blockchain()

print(f"Blockchain created with genesis block: {blockchain.chain[0].hash}")
```

### Step 2: Create a Wallet

```python
from dubchain import Wallet

# Generate a new wallet
wallet = Wallet.generate()

print(f"Wallet address: {wallet.address}")
print(f"Public key: {wallet.public_key}")
```

### Step 3: Create a Transaction

```python
# Create a transaction
transaction = wallet.create_transaction(
    recipient="recipient_address_here",
    amount=1000,  # Amount in smallest unit
    fee=10        # Transaction fee
)

print(f"Transaction created: {transaction.txid}")
```

### Step 4: Add Transaction to Blockchain

```python
# Add transaction to the blockchain
blockchain.add_transaction(transaction)

print(f"Transaction added. Pending transactions: {len(blockchain.pending_transactions)}")
```

### Step 5: Mine a Block

```python
# Mine a new block
block = blockchain.mine_block()

print(f"Block mined: {block.hash}")
print(f"Block height: {block.height}")
print(f"Transactions in block: {len(block.transactions)}")
```

## Smart Contracts

### Step 1: Create a Simple Contract

```python
from dubchain import SmartContract

# Define contract code
contract_code = """
contract SimpleStorage {
    uint256 public storedData;
    
    constructor(uint256 initialValue) {
        storedData = initialValue;
    }
    
    function set(uint256 x) public {
        storedData = x;
    }
    
    function get() public view returns (uint256) {
        return storedData;
    }
}
"""

# Compile the contract
contract = SmartContract.compile(contract_code)
print(f"Contract compiled: {contract.bytecode[:50]}...")
```

### Step 2: Deploy the Contract

```python
# Deploy the contract
deploy_tx = wallet.deploy_contract(contract, constructor_args=[42])
blockchain.add_transaction(deploy_tx)
blockchain.mine_block()

print(f"Contract deployed at: {contract.address}")
```

### Step 3: Interact with the Contract

```python
# Call contract function
call_tx = wallet.call_contract(
    contract.address,
    "set",
    [100]
)
blockchain.add_transaction(call_tx)
blockchain.mine_block()

print("Contract function called successfully")
```

## Consensus Mechanisms

### Proof of Stake (PoS)

```python
from dubchain.consensus import ProofOfStake

# Create PoS consensus
pos = ProofOfStake()

# Add validators
pos.add_validator("validator1", 1000)  # 1000 stake
pos.add_validator("validator2", 2000)  # 2000 stake
pos.add_validator("validator3", 1500)  # 1500 stake

# Select validator for next block
validator = pos.select_validator()
print(f"Selected validator: {validator}")
```

### Delegated Proof of Stake (DPoS)

```python
from dubchain.consensus import DelegatedProofOfStake

# Create DPoS consensus
dpos = DelegatedProofOfStake()

# Add delegates
dpos.add_delegate("delegate1", 5000)  # 5000 votes
dpos.add_delegate("delegate2", 3000)  # 3000 votes
dpos.add_delegate("delegate3", 7000)  # 7000 votes

# Get top delegates
top_delegates = dpos.get_top_delegates(3)
print(f"Top delegates: {top_delegates}")
```

## Cross-Chain Bridge

### Step 1: Configure Bridge

```python
from dubchain.bridge import BridgeManager, BridgeConfig

# Configure bridge
config = BridgeConfig(
    bridge_type="lock_and_mint",
    supported_chains=["ethereum", "bitcoin"],
    supported_assets=["ETH", "BTC"]
)

# Create bridge manager
bridge = BridgeManager(config)
print("Bridge configured successfully")
```

### Step 2: Create Cross-Chain Transaction

```python
# Create cross-chain transaction
cross_chain_tx = bridge.create_cross_chain_transaction(
    source_chain="ethereum",
    target_chain="dubchain",
    source_asset="ETH",
    target_asset="DUB",
    sender="0x1234...",
    receiver="0x5678...",
    amount=1000000000000000000  # 1 ETH in wei
)

print(f"Cross-chain transaction created: {cross_chain_tx.transaction_id}")
```

### Step 3: Process Transaction

```python
# Process the transaction
success = bridge.process_transaction(cross_chain_tx.transaction_id)
print(f"Transaction processed successfully: {success}")
```

## Sharding System

### Step 1: Configure Sharding

```python
from dubchain.sharding import ShardManager, ShardConfig

# Configure sharding
config = ShardConfig(
    max_shards=8,
    min_validators_per_shard=64,
    enable_dynamic_sharding=True
)

# Create shard manager
shard_manager = ShardManager(config)
print("Shard manager created")
```

### Step 2: Create Shards

```python
# Create multiple shards
shards = []
for i in range(4):
    shard = shard_manager.create_shard()
    shards.append(shard)
    print(f"Created shard {shard.shard_id}")

print(f"Total shards: {len(shards)}")
```

### Step 3: Manage Shard State

```python
# Get shard information
for shard in shards:
    print(f"Shard {shard.shard_id}:")
    print(f"  Status: {shard.status}")
    print(f"  Validators: {shard.metrics.validator_count}")
    print(f"  Transactions: {shard.metrics.transaction_count}")
```

## Networking

### Step 1: Start a Node

```python
from dubchain.network import Peer, NetworkTopology

# Create a peer
peer = Peer("node1", "127.0.0.1", 8080)

# Start the peer
peer.start()
print(f"Peer started on {peer.host}:{peer.port}")
```

### Step 2: Connect to Network

```python
# Connect to other peers
peer.connect_to("127.0.0.1", 8081)
peer.connect_to("127.0.0.1", 8082)

print(f"Connected to {len(peer.connections)} peers")
```

### Step 3: Send Messages

```python
# Send a message to all connected peers
message = {
    "type": "block",
    "data": {"height": 1, "hash": "abc123..."}
}

peer.broadcast(message)
print("Message broadcasted to network")
```

## Complete Example

Here's a complete example that demonstrates multiple DubChain features:

```python
#!/usr/bin/env python3
"""
DubChain Complete Example
Demonstrates blockchain creation, transactions, smart contracts, and consensus.
"""

from dubchain import Blockchain, Wallet, SmartContract
from dubchain.consensus import ProofOfStake
from dubchain.bridge import BridgeManager, BridgeConfig
from dubchain.sharding import ShardManager, ShardConfig

def main():
    print("=== DubChain Complete Example ===\n")
    
    # 1. Create blockchain with PoS consensus
    print("1. Creating blockchain with Proof of Stake consensus...")
    blockchain = Blockchain()
    pos = ProofOfStake()
    
    # Add validators
    pos.add_validator("validator1", 1000)
    pos.add_validator("validator2", 2000)
    pos.add_validator("validator3", 1500)
    
    print(f"   Blockchain created with {len(pos.validators)} validators")
    
    # 2. Create wallets
    print("\n2. Creating wallets...")
    alice = Wallet.generate()
    bob = Wallet.generate()
    charlie = Wallet.generate()
    
    print(f"   Alice: {alice.address}")
    print(f"   Bob: {bob.address}")
    print(f"   Charlie: {charlie.address}")
    
    # 3. Create and process transactions
    print("\n3. Creating transactions...")
    
    # Alice sends to Bob
    tx1 = alice.create_transaction(bob.address, 1000, 10)
    blockchain.add_transaction(tx1)
    
    # Bob sends to Charlie
    tx2 = bob.create_transaction(charlie.address, 500, 10)
    blockchain.add_transaction(tx2)
    
    print(f"   Created {len(blockchain.pending_transactions)} transactions")
    
    # 4. Mine block
    print("\n4. Mining block...")
    block = blockchain.mine_block()
    print(f"   Block mined: {block.hash[:20]}...")
    print(f"   Block height: {block.height}")
    
    # 5. Deploy smart contract
    print("\n5. Deploying smart contract...")
    contract_code = """
    contract SimpleToken {
        mapping(address => uint256) public balances;
        
        constructor() {
            balances[msg.sender] = 1000000;
        }
        
        function transfer(address to, uint256 amount) public {
            require(balances[msg.sender] >= amount);
            balances[msg.sender] -= amount;
            balances[to] += amount;
        }
    }
    """
    
    contract = SmartContract.compile(contract_code)
    deploy_tx = alice.deploy_contract(contract)
    blockchain.add_transaction(deploy_tx)
    blockchain.mine_block()
    
    print(f"   Contract deployed at: {contract.address}")
    
    # 6. Configure cross-chain bridge
    print("\n6. Configuring cross-chain bridge...")
    bridge_config = BridgeConfig(
        bridge_type="lock_and_mint",
        supported_chains=["ethereum", "bitcoin"],
        supported_assets=["ETH", "BTC"]
    )
    bridge = BridgeManager(bridge_config)
    print("   Bridge configured successfully")
    
    # 7. Configure sharding
    print("\n7. Configuring sharding system...")
    shard_config = ShardConfig(
        max_shards=4,
        min_validators_per_shard=32,
        enable_dynamic_sharding=True
    )
    shard_manager = ShardManager(shard_config)
    
    # Create shards
    for i in range(2):
        shard = shard_manager.create_shard()
        print(f"   Created shard {shard.shard_id}")
    
    # 8. Display final state
    print("\n8. Final blockchain state:")
    print(f"   Chain height: {len(blockchain.chain)}")
    print(f"   Total transactions: {sum(len(block.transactions) for block in blockchain.chain)}")
    print(f"   Pending transactions: {len(blockchain.pending_transactions)}")
    print(f"   Active shards: {len(shard_manager.shards)}")
    
    print("\n=== Example completed successfully! ===")

if __name__ == "__main__":
    main()
```

## Next Steps

Now that you've completed the quick start guide:

1. **Explore Examples**: Run the example scripts in the `examples/` directory
2. **Read Documentation**: Dive deeper into specific modules
3. **Experiment**: Try modifying the examples and creating your own
4. **Join the Community**: Participate in discussions and contribute

### Recommended Learning Path

1. **Core Concepts**: Understand blockchain fundamentals
2. **Basic Operations**: Master transaction and block operations
3. **Smart Contracts**: Learn contract development and deployment
4. **Advanced Features**: Explore consensus, bridging, and sharding
5. **Research Applications**: Use DubChain for your research projects

### Additional Resources

- **[Architecture Documentation](architecture/README.md)**: Deep dive into system design
- **[Module Documentation](modules/README.md)**: Detailed module documentation
- **[API Reference](api/README.md)**: Complete API documentation
- **[Tutorials](tutorials/README.md)**: Step-by-step tutorials
- **[Research Papers](research/README.md)**: Academic research and findings

## Getting Help

If you need help:

1. **Check the documentation**: Most questions are answered in the docs
2. **Run examples**: The examples demonstrate common use cases
3. **Search issues**: Check existing GitHub issues
4. **Ask questions**: Create a GitHub issue or discussion
5. **Join discussions**: Participate in the community

Welcome to DubChain! We're excited to see what you'll build.
