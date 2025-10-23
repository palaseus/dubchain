"""
GraphQL Schema Definition for DubChain API.

This module defines the complete GraphQL schema including queries, mutations,
and subscriptions for blockchain operations, smart contracts, wallets, bridges,
sharding, consensus, and governance.
"""

import logging

logger = logging.getLogger(__name__)
import strawberry
from typing import List, Optional, Union
from datetime import datetime
from decimal import Decimal

# Core blockchain types
@strawberry.type
class Block:
    """Blockchain block representation."""
    hash: str
    previous_hash: str
    timestamp: datetime
    height: int
    transactions: List["Transaction"]
    merkle_root: str
    nonce: int
    difficulty: int
    validator: str
    gas_used: int
    gas_limit: int

@strawberry.type
class Transaction:
    """Blockchain transaction representation."""
    hash: str
    from_address: str
    to_address: str
    value: Decimal
    gas_price: int
    gas_limit: int
    gas_used: int
    nonce: int
    data: Optional[str]
    signature: str
    status: str
    block_height: Optional[int]
    timestamp: datetime

@strawberry.type
class Account:
    """Account/address representation."""
    address: str
    balance: Decimal
    nonce: int
    code_hash: Optional[str]
    storage_root: Optional[str]
    transaction_count: int
    is_contract: bool

# Smart contract types
@strawberry.type
class Contract:
    """Smart contract representation."""
    address: str
    name: str
    abi: str
    bytecode: str
    source_code: Optional[str]
    creator: str
    created_at: datetime
    gas_used: int
    storage_size: int

@strawberry.type
class ContractEvent:
    """Smart contract event."""
    contract_address: str
    event_name: str
    event_data: str
    block_height: int
    transaction_hash: str
    log_index: int
    timestamp: datetime

# Wallet types
@strawberry.type
class Wallet:
    """Wallet representation."""
    wallet_id: str
    name: str
    wallet_type: str
    accounts: List[Account]
    created_at: datetime
    last_accessed: datetime
    is_encrypted: bool

@strawberry.type
class WalletAccount:
    """Wallet account representation."""
    account_index: int
    address: str
    public_key: str
    balance: Decimal
    transaction_count: int
    last_used: Optional[datetime]
    label: Optional[str]

# Bridge types
@strawberry.type
class BridgeTransfer:
    """Cross-chain bridge transfer."""
    transfer_id: str
    source_chain: str
    target_chain: str
    source_asset: str
    target_asset: str
    amount: Decimal
    sender: str
    receiver: str
    status: str
    created_at: datetime
    completed_at: Optional[datetime]
    transaction_hash: Optional[str]

@strawberry.type
class BridgeAsset:
    """Bridge asset representation."""
    asset_id: str
    symbol: str
    name: str
    decimals: int
    total_supply: Decimal
    circulating_supply: Decimal
    supported_chains: List[str]

# Sharding types
@strawberry.type
class Shard:
    """Network shard representation."""
    shard_id: int
    validator_count: int
    transaction_count: int
    block_height: int
    status: str
    validators: List[str]
    cross_shard_transactions: int

@strawberry.type
class CrossShardTransaction:
    """Cross-shard transaction."""
    transaction_id: str
    source_shard: int
    target_shard: int
    amount: Decimal
    sender: str
    receiver: str
    status: str
    created_at: datetime
    completed_at: Optional[datetime]

# Consensus types
@strawberry.type
class Validator:
    """Validator representation."""
    address: str
    public_key: str
    stake: Decimal
    commission_rate: float
    status: str
    uptime: float
    blocks_proposed: int
    blocks_validated: int
    slashing_events: int
    last_active: datetime

@strawberry.type
class Delegation:
    """Delegation representation."""
    delegator: str
    validator: str
    amount: Decimal
    shares: Decimal
    created_at: datetime
    unbonding_time: Optional[datetime]

# Governance types
@strawberry.type
class Proposal:
    """Governance proposal."""
    proposal_id: str
    proposer: str
    title: str
    description: str
    proposal_type: str
    status: str
    voting_start: datetime
    voting_end: datetime
    execution_time: Optional[datetime]
    total_votes: int
    yes_votes: int
    no_votes: int
    abstain_votes: int
    quorum: float
    threshold: float

@strawberry.type
class Vote:
    """Governance vote."""
    proposal_id: str
    voter: str
    vote_option: str
    voting_power: Decimal
    timestamp: datetime
    reason: Optional[str]

# Network types
@strawberry.type
class Peer:
    """Network peer representation."""
    peer_id: str
    address: str
    port: int
    public_key: str
    connection_type: str
    last_seen: datetime
    latency: Optional[float]
    is_active: bool

@strawberry.type
class NetworkStats:
    """Network statistics."""
    total_peers: int
    active_peers: int
    total_connections: int
    bytes_sent: int
    bytes_received: int
    messages_sent: int
    messages_received: int
    average_latency: float

# Performance types
@strawberry.type
class PerformanceMetrics:
    """Performance metrics."""
    tps: float
    latency: float
    gas_efficiency: float
    memory_usage: int
    cpu_usage: float
    disk_usage: int
    network_throughput: float
    timestamp: datetime

# Query types
@strawberry.type
class Query:
    """Root query type."""
    
    @strawberry.field
    def block(self, hash: Optional[str] = None, height: Optional[int] = None) -> Optional[Block]:
        """Get block by hash or height."""
        # Implementation will be in resolvers
        pass
    
    @strawberry.field
    def blocks(self, limit: int = 10, offset: int = 0) -> List[Block]:
        """Get recent blocks."""
        # Implementation will be in resolvers
        pass
    
    @strawberry.field
    def transaction(self, hash: str) -> Optional[Transaction]:
        """Get transaction by hash."""
        # Implementation will be in resolvers
        pass
    
    @strawberry.field
    def account(self, address: str) -> Optional[Account]:
        """Get account by address."""
        # Implementation will be in resolvers
        pass
    
    @strawberry.field
    def contract(self, address: str) -> Optional[Contract]:
        """Get contract by address."""
        # Implementation will be in resolvers
        pass
    
    @strawberry.field
    def wallet(self, wallet_id: str) -> Optional[Wallet]:
        """Get wallet by ID."""
        # Implementation will be in resolvers
        pass
    
    @strawberry.field
    def bridge_transfer(self, transfer_id: str) -> Optional[BridgeTransfer]:
        """Get bridge transfer by ID."""
        # Implementation will be in resolvers
        pass
    
    @strawberry.field
    def shard(self, shard_id: int) -> Optional[Shard]:
        """Get shard by ID."""
        # Implementation will be in resolvers
        pass
    
    @strawberry.field
    def validator(self, address: str) -> Optional[Validator]:
        """Get validator by address."""
        # Implementation will be in resolvers
        pass
    
    @strawberry.field
    def proposal(self, proposal_id: str) -> Optional[Proposal]:
        """Get proposal by ID."""
        # Implementation will be in resolvers
        pass
    
    @strawberry.field
    def network_stats(self) -> NetworkStats:
        """Get network statistics."""
        # Implementation will be in resolvers
        pass
    
    @strawberry.field
    def performance_metrics(self) -> PerformanceMetrics:
        """Get performance metrics."""
        # Implementation will be in resolvers
        pass

# Mutation types
@strawberry.type
class Mutation:
    """Root mutation type."""
    
    @strawberry.field
    def create_wallet(self, name: str, password: Optional[str] = None) -> str:
        """Create a new wallet."""
        # Implementation will be in resolvers
        pass
    
    @strawberry.field
    def create_transaction(self, 
                          from_address: str,
                          to_address: str,
                          value: Decimal,
                          gas_price: int,
                          gas_limit: int,
                          data: Optional[str] = None) -> str:
        """Create a new transaction."""
        # Implementation will be in resolvers
        pass
    
    @strawberry.field
    def deploy_contract(self,
                       bytecode: str,
                       abi: str,
                       constructor_args: Optional[List[str]] = None) -> str:
        """Deploy a smart contract."""
        # Implementation will be in resolvers
        pass
    
    @strawberry.field
    def call_contract(self,
                     contract_address: str,
                     function_name: str,
                     args: Optional[List[str]] = None) -> str:
        """Call a smart contract function."""
        # Implementation will be in resolvers
        pass
    
    @strawberry.field
    def create_bridge_transfer(self,
                              source_chain: str,
                              target_chain: str,
                              source_asset: str,
                              target_asset: str,
                              amount: Decimal,
                              receiver: str) -> str:
        """Create a bridge transfer."""
        # Implementation will be in resolvers
        pass
    
    @strawberry.field
    def create_proposal(self,
                       title: str,
                       description: str,
                       proposal_type: str) -> str:
        """Create a governance proposal."""
        # Implementation will be in resolvers
        pass
    
    @strawberry.field
    def vote(self,
            proposal_id: str,
            vote_option: str,
            reason: Optional[str] = None) -> bool:
        """Vote on a proposal."""
        # Implementation will be in resolvers
        pass

# Subscription types
@strawberry.type
class Subscription:
    """Root subscription type."""
    
    @strawberry.subscription
    async def new_blocks(self) -> Block:
        """Subscribe to new blocks."""
        # Implementation will be in resolvers
        pass
    
    @strawberry.subscription
    async def new_transactions(self) -> Transaction:
        """Subscribe to new transactions."""
        # Implementation will be in resolvers
        pass
    
    @strawberry.subscription
    async def contract_events(self, contract_address: str) -> ContractEvent:
        """Subscribe to contract events."""
        # Implementation will be in resolvers
        pass
    
    @strawberry.subscription
    async def bridge_transfer_updates(self, transfer_id: str) -> BridgeTransfer:
        """Subscribe to bridge transfer updates."""
        # Implementation will be in resolvers
        pass
    
    @strawberry.subscription
    async def shard_updates(self, shard_id: int) -> Shard:
        """Subscribe to shard updates."""
        # Implementation will be in resolvers
        pass
    
    @strawberry.subscription
    async def governance_updates(self) -> Proposal:
        """Subscribe to governance updates."""
        # Implementation will be in resolvers
        pass
    
    @strawberry.subscription
    async def network_stats_updates(self) -> NetworkStats:
        """Subscribe to network statistics updates."""
        # Implementation will be in resolvers
        pass
    
    @strawberry.subscription
    async def performance_metrics_updates(self) -> PerformanceMetrics:
        """Subscribe to performance metrics updates."""
        # Implementation will be in resolvers
        pass

# Create the schema
schema = strawberry.Schema(
    query=Query,
    mutation=Mutation,
    subscription=Subscription
)
