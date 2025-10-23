# DubChain API Reference

Complete API documentation for all DubChain modules.

## Core

- **[Blockchain](blockchain.md)** - `dubchain.core.blockchain`
  - Main blockchain implementation
- **[Block](block.md)** - `dubchain.core.block`
  - A blockchain block containing transactions
- **[Transaction](transaction.md)** - `dubchain.core.transaction`
  - A blockchain transaction
- **[Consensus](consensus.md)** - `dubchain.core.consensus`
  - Configuration for consensus mechanisms

## Consensus

- **[Proof Of Stake](proof_of_stake.md)** - `dubchain.consensus.proof_of_stake`
  - Proof of Stake consensus implementation
- **[Pbft](pbft.md)** - `dubchain.consensus.pbft`
  - PBFT consensus implementation
- **[Consensus Types](consensus_types.md)** - `dubchain.consensus.consensus_types`
  - Configuration for consensus mechanisms

## Cryptography

- **[Signatures](signatures.md)** - `dubchain.crypto.signatures`
  - ECDSA signature operations with secp256k1 curve
- **[Hashing](hashing.md)** - `dubchain.crypto.hashing`
  - Immutable hash value with comparison and string representation
- **[Merkle](merkle.md)** - `dubchain.crypto.merkle`
  - Proof of inclusion in a Merkle tree

## Virtual Machine

- **[Execution Engine](execution_engine.md)** - `dubchain.vm.execution_engine`
  - Execution context for a contract call
- **[Gas Meter](gas_meter.md)** - `dubchain.vm.gas_meter`
  - Simple gas cost constants for compatibility
- **[Opcodes](opcodes.md)** - `dubchain.vm.opcodes`
  - Simple opcode representation for compatibility

## Network

- **[Peer](peer.md)** - `dubchain.network.peer`
  - Types of peer connections
- **[Protocol](protocol.md)** - `dubchain.network.protocol`
  - Routes messages to appropriate handlers

## Sharding

- **[Shard Manager](shard_manager.md)** - `dubchain.sharding.shard_manager`
  - Allocates validators to shards
- **[Shard Types](shard_types.md)** - `dubchain.sharding.shard_types`
  - Cross-shard transaction data

## State Channels

- **[Channel](channel.md)** - `dubchain.state_channels.channel`
  - A participant in a state channel
- **[Channel Protocol](channel_protocol.md)** - `dubchain.state_channels.channel_protocol`
  - Reasons for channel closure

## Governance

- **[Core](core.md)** - `dubchain.governance.core`
  - Configuration for the governance system
- **[Proposal](proposal.md)** - `dubchain.governance.proposal`
  - Manages governance proposals and voting

## Quick Start

```python
from dubchain import Blockchain, PrivateKey, PublicKey

# Create blockchain
blockchain = Blockchain()
private_key = PrivateKey.generate()
public_key = private_key.get_public_key()
blockchain.create_genesis_block(public_key.to_address())

# Create transaction
tx = blockchain.create_transfer_transaction(
    sender_private_key=private_key,
    recipient_address=public_key.to_address(),
    amount=1000,
    fee=10
)

# Add transaction and mine block
blockchain.add_transaction(tx)
block = blockchain.mine_block(public_key.to_address())
```
