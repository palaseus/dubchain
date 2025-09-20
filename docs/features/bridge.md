# Cross-Chain Bridge

This document explains the cross-chain bridge implementation in DubChain.

## Overview

The cross-chain bridge enables interoperability between DubChain and other blockchain networks, allowing users to transfer assets and data across different chains.

## Bridge Architecture

### Components

1. **Bridge Manager**: Coordinates bridge operations
2. **Atomic Swaps**: Trustless cross-chain exchanges
3. **Cross-Chain Messaging**: Inter-chain communication
4. **Universal Assets**: Asset representation across chains
5. **Bridge Security**: Fraud detection and validation

### Bridge Types

#### Lock and Mint
- Lock assets on source chain
- Mint equivalent assets on target chain
- Burn assets on target chain to unlock on source

#### Atomic Swaps
- Trustless peer-to-peer exchanges
- Time-locked transactions
- Hash-locked contracts

#### Relayed Bridges
- Trusted relayers validate transactions
- Faster finality
- Centralized trust model

## Implementation

### Bridge Manager
```python
class BridgeManager:
    def __init__(self, config):
        self.config = config
        self.supported_chains = config.supported_chains
        self.supported_assets = config.supported_assets
        self.relayers = []
        self.pending_transactions = {}
    
    def create_cross_chain_transaction(self, source_chain, target_chain, 
                                     source_asset, target_asset, 
                                     sender, receiver, amount):
        """Create cross-chain transaction."""
        transaction = CrossChainTransaction(
            source_chain=source_chain,
            target_chain=target_chain,
            source_asset=source_asset,
            target_asset=target_asset,
            sender=sender,
            receiver=receiver,
            amount=amount,
            timestamp=time.time()
        )
        
        self.pending_transactions[transaction.id] = transaction
        return transaction
    
    def process_transaction(self, transaction_id):
        """Process cross-chain transaction."""
        if transaction_id not in self.pending_transactions:
            return False
        
        transaction = self.pending_transactions[transaction_id]
        
        # Validate transaction
        if not self.validate_transaction(transaction):
            return False
        
        # Execute bridge logic
        success = self.execute_bridge_transaction(transaction)
        
        if success:
            del self.pending_transactions[transaction_id]
        
        return success
```

### Atomic Swaps
```python
class AtomicSwap:
    def __init__(self, initiator, responder, asset1, asset2, amount1, amount2):
        self.initiator = initiator
        self.responder = responder
        self.asset1 = asset1
        self.asset2 = asset2
        self.amount1 = amount1
        self.amount2 = amount2
        self.secret_hash = None
        self.secret = None
        self.timeout = 24 * 60 * 60  # 24 hours
        self.status = "pending"
    
    def initiate_swap(self):
        """Initiate atomic swap."""
        # Generate secret and hash
        self.secret = generate_random_secret()
        self.secret_hash = hash_secret(self.secret)
        
        # Create time-locked transaction
        tx1 = create_timelocked_transaction(
            sender=self.initiator,
            recipient=self.responder,
            asset=self.asset1,
            amount=self.amount1,
            secret_hash=self.secret_hash,
            timeout=self.timeout
        )
        
        return tx1
    
    def complete_swap(self, secret):
        """Complete atomic swap with secret."""
        if hash_secret(secret) != self.secret_hash:
            return False
        
        # Create second transaction
        tx2 = create_transaction(
            sender=self.responder,
            recipient=self.initiator,
            asset=self.asset2,
            amount=self.amount2
        )
        
        self.status = "completed"
        return tx2
```

### Cross-Chain Messaging
```python
class CrossChainMessaging:
    def __init__(self, bridge_manager):
        self.bridge_manager = bridge_manager
        self.message_queue = []
        self.message_handlers = {}
    
    def send_message(self, target_chain, message_type, payload):
        """Send message to target chain."""
        message = CrossChainMessage(
            source_chain=self.bridge_manager.chain_id,
            target_chain=target_chain,
            message_type=message_type,
            payload=payload,
            timestamp=time.time()
        )
        
        # Sign message
        message.signature = self.sign_message(message)
        
        # Add to queue
        self.message_queue.append(message)
        
        return message.id
    
    def process_message(self, message):
        """Process incoming cross-chain message."""
        # Verify message signature
        if not self.verify_message_signature(message):
            return False
        
        # Check message handler
        if message.message_type in self.message_handlers:
            handler = self.message_handlers[message.message_type]
            return handler(message.payload)
        
        return False
    
    def register_handler(self, message_type, handler):
        """Register message handler."""
        self.message_handlers[message_type] = handler
```

### Universal Assets
```python
class UniversalAsset:
    def __init__(self, asset_id, name, symbol, decimals, chains):
        self.asset_id = asset_id
        self.name = name
        self.symbol = symbol
        self.decimals = decimals
        self.chains = chains
        self.total_supply = {}
        self.circulating_supply = {}
    
    def mint_on_chain(self, chain_id, amount):
        """Mint asset on specific chain."""
        if chain_id not in self.chains:
            return False
        
        if chain_id not in self.total_supply:
            self.total_supply[chain_id] = 0
        
        self.total_supply[chain_id] += amount
        return True
    
    def burn_on_chain(self, chain_id, amount):
        """Burn asset on specific chain."""
        if chain_id not in self.chains:
            return False
        
        if chain_id not in self.total_supply:
            return False
        
        if self.total_supply[chain_id] < amount:
            return False
        
        self.total_supply[chain_id] -= amount
        return True
    
    def get_total_supply(self):
        """Get total supply across all chains."""
        return sum(self.total_supply.values())
```

## Security Mechanisms

### Fraud Detection
```python
class FraudDetector:
    def __init__(self):
        self.suspicious_patterns = []
        self.alert_threshold = 0.8
    
    def analyze_transaction(self, transaction):
        """Analyze transaction for fraud indicators."""
        risk_score = 0.0
        
        # Check for unusual amounts
        if transaction.amount > self.get_average_amount() * 10:
            risk_score += 0.3
        
        # Check for rapid transactions
        if self.is_rapid_transaction(transaction.sender):
            risk_score += 0.2
        
        # Check for known malicious addresses
        if transaction.sender in self.get_malicious_addresses():
            risk_score += 0.5
        
        return risk_score
    
    def should_block_transaction(self, transaction):
        """Determine if transaction should be blocked."""
        risk_score = self.analyze_transaction(transaction)
        return risk_score > self.alert_threshold
```

### Multi-Signature Validation
```python
class MultiSigValidator:
    def __init__(self, required_signatures):
        self.required_signatures = required_signatures
        self.validators = []
    
    def add_validator(self, validator_address):
        """Add validator to multi-sig."""
        self.validators.append(validator_address)
    
    def validate_transaction(self, transaction, signatures):
        """Validate transaction with multiple signatures."""
        valid_signatures = 0
        
        for signature in signatures:
            if self.verify_signature(transaction, signature):
                valid_signatures += 1
        
        return valid_signatures >= self.required_signatures
```

## Performance Optimization

### Batch Processing
```python
class BatchProcessor:
    def __init__(self, batch_size=100):
        self.batch_size = batch_size
        self.pending_transactions = []
    
    def add_transaction(self, transaction):
        """Add transaction to batch."""
        self.pending_transactions.append(transaction)
        
        if len(self.pending_transactions) >= self.batch_size:
            self.process_batch()
    
    def process_batch(self):
        """Process batch of transactions."""
        if not self.pending_transactions:
            return
        
        # Process transactions in parallel
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            for transaction in self.pending_transactions:
                future = executor.submit(self.process_single_transaction, transaction)
                futures.append(future)
            
            # Wait for all transactions to complete
            results = [future.result() for future in futures]
        
        # Clear processed transactions
        self.pending_transactions.clear()
        
        return results
```

### Caching
```python
class BridgeCache:
    def __init__(self, max_size=10000):
        self.cache = {}
        self.max_size = max_size
    
    def get_cached_balance(self, address, asset):
        """Get cached balance."""
        key = (address, asset)
        return self.cache.get(key)
    
    def cache_balance(self, address, asset, balance):
        """Cache balance."""
        if len(self.cache) >= self.max_size:
            # Remove oldest entry
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        
        key = (address, asset)
        self.cache[key] = balance
```

## Usage Examples

### Basic Bridge Usage
```python
# Create bridge manager
bridge_config = BridgeConfig(
    bridge_type="lock_and_mint",
    supported_chains=["ethereum", "bitcoin", "dubchain"],
    supported_assets=["ETH", "BTC", "DUB"]
)
bridge = BridgeManager(bridge_config)

# Create cross-chain transaction
tx = bridge.create_cross_chain_transaction(
    source_chain="ethereum",
    target_chain="dubchain",
    source_asset="ETH",
    target_asset="DUB",
    sender="0x...",
    receiver="0x...",
    amount=1000000000000000000  # 1 ETH
)

# Process transaction
success = bridge.process_transaction(tx.id)
print(f"Cross-chain transaction successful: {success}")
```

### Atomic Swap Usage
```python
# Create atomic swap
swap = AtomicSwap(
    initiator="alice",
    responder="bob",
    asset1="ETH",
    asset2="BTC",
    amount1=1000000000000000000,  # 1 ETH
    amount2=25000000  # 0.25 BTC
)

# Initiate swap
tx1 = swap.initiate_swap()
print(f"Initiated swap with transaction: {tx1.id}")

# Complete swap
tx2 = swap.complete_swap(secret)
print(f"Completed swap with transaction: {tx2.id}")
```

## Security Best Practices

1. **Multi-signature validation** for critical operations
2. **Time locks** to prevent rapid withdrawals
3. **Fraud detection** systems
4. **Regular security audits**
5. **Emergency pause mechanisms**
6. **Decentralized governance**

## Further Reading

- [Blockchain Fundamentals](../concepts/blockchain.md)
- [Consensus Mechanisms](../concepts/consensus.md)
- [Cryptography](../concepts/cryptography.md)
- [Performance Optimization](../performance/README.md)
