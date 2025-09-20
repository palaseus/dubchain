# Configuration Guide

This document explains how to configure DubChain for different environments and use cases.

## Configuration Files

### Main Configuration
- `config.yaml`: Main configuration file
- `network.yaml`: Network-specific settings
- `consensus.yaml`: Consensus mechanism settings

### Environment-Specific
- `development.yaml`: Development environment
- `staging.yaml`: Staging environment
- `production.yaml`: Production environment

## Configuration Options

### Blockchain Settings
```yaml
blockchain:
  block_time: 10  # seconds
  max_block_size: 1048576  # bytes
  max_transactions_per_block: 1000
  difficulty_adjustment_interval: 2016
```

### Consensus Settings
```yaml
consensus:
  mechanism: "proof_of_stake"
  validators:
    min_stake: 1000000
    max_validators: 100
  block_reward: 50
```

### Network Settings
```yaml
network:
  port: 8080
  max_peers: 50
  discovery:
    bootstrap_nodes:
      - "127.0.0.1:8080"
      - "127.0.0.1:8081"
```

## Usage Examples

### Loading Configuration
```python
from dubchain.config import Config

# Load configuration
config = Config.load("config.yaml")

# Access configuration values
block_time = config.blockchain.block_time
max_peers = config.network.max_peers
```

### Environment-Specific Configuration
```python
import os

# Load environment-specific config
env = os.getenv("DUBCHAIN_ENV", "development")
config = Config.load(f"{env}.yaml")
```

## Further Reading

- [Installation Guide](../installation/README.md)
- [Quick Start Guide](../quickstart/README.md)
