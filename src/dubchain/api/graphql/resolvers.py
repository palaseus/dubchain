"""
GraphQL Resolvers for DubChain API.

This module implements all GraphQL resolvers for queries, mutations, and subscriptions.
"""

import asyncio
import json
from datetime import datetime
from decimal import Decimal
from typing import List, Optional, AsyncGenerator

import strawberry
from strawberry.types import Info

from ..core.blockchain import Blockchain
from ..core.transaction import Transaction as CoreTransaction
from ..vm.contract import SmartContract
from ..wallet.wallet_manager import WalletManager
from ..bridge.bridge_manager import BridgeManager
from ..sharding.shard_manager import ShardManager
from ..consensus.consensus_engine import ConsensusEngine
from ..governance.core import GovernanceEngine
from ..network.peer import PeerManager
from ..performance.monitoring import PerformanceMonitor

from .schema import (
    Block, Transaction, Account, Contract, ContractEvent, Wallet, WalletAccount,
    BridgeTransfer, BridgeAsset, Shard, CrossShardTransaction, Validator,
    Delegation, Proposal, Vote, Peer, NetworkStats, PerformanceMetrics
)

# Global instances (would be injected in production)
blockchain = Blockchain()
wallet_manager = WalletManager()
bridge_manager = BridgeManager()
shard_manager = ShardManager()
consensus_engine = ConsensusEngine()
governance_engine = GovernanceEngine()
peer_manager = PeerManager()
performance_monitor = PerformanceMonitor()

class QueryResolvers:
    """Query resolvers implementation."""
    
    @staticmethod
    def block(hash: Optional[str] = None, height: Optional[int] = None) -> Optional[Block]:
        """Get block by hash or height."""
        try:
            if hash:
                core_block = blockchain.get_block_by_hash(hash)
            elif height is not None:
                core_block = blockchain.get_block_by_height(height)
            else:
                return None
            
            if not core_block:
                return None
            
            # Convert core block to GraphQL block
            transactions = []
            for tx in core_block.transactions:
                transactions.append(Transaction(
                    hash=tx.hash,
                    from_address=tx.sender,
                    to_address=tx.recipient,
                    value=Decimal(str(tx.amount)),
                    gas_price=tx.gas_price,
                    gas_limit=tx.gas_limit,
                    gas_used=tx.gas_used,
                    nonce=tx.nonce,
                    data=tx.data.hex() if tx.data else None,
                    signature=tx.signature.hex(),
                    status="confirmed",
                    block_height=core_block.height,
                    timestamp=datetime.fromtimestamp(core_block.timestamp)
                ))
            
            return Block(
                hash=core_block.hash,
                previous_hash=core_block.previous_hash,
                timestamp=datetime.fromtimestamp(core_block.timestamp),
                height=core_block.height,
                transactions=transactions,
                merkle_root=core_block.merkle_root,
                nonce=core_block.nonce,
                difficulty=core_block.difficulty,
                validator=core_block.validator,
                gas_used=core_block.gas_used,
                gas_limit=core_block.gas_limit
            )
        except Exception as e:
            print(f"Error getting block: {e}")
            return None
    
    @staticmethod
    def blocks(limit: int = 10, offset: int = 0) -> List[Block]:
        """Get recent blocks."""
        try:
            blocks = []
            current_height = blockchain.get_latest_height()
            
            for i in range(offset, min(offset + limit, current_height + 1)):
                block = QueryResolvers.block(height=current_height - i)
                if block:
                    blocks.append(block)
            
            return blocks
        except Exception as e:
            print(f"Error getting blocks: {e}")
            return []
    
    @staticmethod
    def transaction(hash: str) -> Optional[Transaction]:
        """Get transaction by hash."""
        try:
            core_tx = blockchain.get_transaction(hash)
            if not core_tx:
                return None
            
            return Transaction(
                hash=core_tx.hash,
                from_address=core_tx.sender,
                to_address=core_tx.recipient,
                value=Decimal(str(core_tx.amount)),
                gas_price=core_tx.gas_price,
                gas_limit=core_tx.gas_limit,
                gas_used=core_tx.gas_used,
                nonce=core_tx.nonce,
                data=core_tx.data.hex() if core_tx.data else None,
                signature=core_tx.signature.hex(),
                status="confirmed" if core_tx.block_height else "pending",
                block_height=core_tx.block_height,
                timestamp=datetime.fromtimestamp(core_tx.timestamp)
            )
        except Exception as e:
            print(f"Error getting transaction: {e}")
            return None
    
    @staticmethod
    def account(address: str) -> Optional[Account]:
        """Get account by address."""
        try:
            account_data = blockchain.get_account(address)
            if not account_data:
                return None
            
            return Account(
                address=address,
                balance=Decimal(str(account_data.get('balance', 0))),
                nonce=account_data.get('nonce', 0),
                code_hash=account_data.get('code_hash'),
                storage_root=account_data.get('storage_root'),
                transaction_count=account_data.get('transaction_count', 0),
                is_contract=account_data.get('is_contract', False)
            )
        except Exception as e:
            print(f"Error getting account: {e}")
            return None
    
    @staticmethod
    def contract(address: str) -> Optional[Contract]:
        """Get contract by address."""
        try:
            contract_data = blockchain.get_contract(address)
            if not contract_data:
                return None
            
            return Contract(
                address=address,
                name=contract_data.get('name', 'Unknown'),
                abi=json.dumps(contract_data.get('abi', {})),
                bytecode=contract_data.get('bytecode', ''),
                source_code=contract_data.get('source_code'),
                creator=contract_data.get('creator', ''),
                created_at=datetime.fromtimestamp(contract_data.get('created_at', 0)),
                gas_used=contract_data.get('gas_used', 0),
                storage_size=contract_data.get('storage_size', 0)
            )
        except Exception as e:
            print(f"Error getting contract: {e}")
            return None
    
    @staticmethod
    def wallet(wallet_id: str) -> Optional[Wallet]:
        """Get wallet by ID."""
        try:
            wallet_data = wallet_manager.get_wallet_info(wallet_id)
            if not wallet_data:
                return None
            
            # Get accounts for this wallet
            accounts = []
            for account_data in wallet_data.get('accounts', []):
                accounts.append(WalletAccount(
                    account_index=account_data.get('account_index', 0),
                    address=account_data.get('address', ''),
                    public_key=account_data.get('public_key', ''),
                    balance=Decimal(str(account_data.get('balance', 0))),
                    transaction_count=account_data.get('transaction_count', 0),
                    last_used=datetime.fromtimestamp(account_data.get('last_used', 0)) if account_data.get('last_used') else None,
                    label=account_data.get('label')
                ))
            
            return Wallet(
                wallet_id=wallet_id,
                name=wallet_data.get('name', ''),
                wallet_type=wallet_data.get('wallet_type', ''),
                accounts=accounts,
                created_at=datetime.fromtimestamp(wallet_data.get('created_at', 0)),
                last_accessed=datetime.fromtimestamp(wallet_data.get('last_accessed', 0)),
                is_encrypted=wallet_data.get('is_encrypted', False)
            )
        except Exception as e:
            print(f"Error getting wallet: {e}")
            return None
    
    @staticmethod
    def bridge_transfer(transfer_id: str) -> Optional[BridgeTransfer]:
        """Get bridge transfer by ID."""
        try:
            transfer_data = bridge_manager.get_transfer(transfer_id)
            if not transfer_data:
                return None
            
            return BridgeTransfer(
                transfer_id=transfer_id,
                source_chain=transfer_data.get('source_chain', ''),
                target_chain=transfer_data.get('target_chain', ''),
                source_asset=transfer_data.get('source_asset', ''),
                target_asset=transfer_data.get('target_asset', ''),
                amount=Decimal(str(transfer_data.get('amount', 0))),
                sender=transfer_data.get('sender', ''),
                receiver=transfer_data.get('receiver', ''),
                status=transfer_data.get('status', 'pending'),
                created_at=datetime.fromtimestamp(transfer_data.get('created_at', 0)),
                completed_at=datetime.fromtimestamp(transfer_data.get('completed_at', 0)) if transfer_data.get('completed_at') else None,
                transaction_hash=transfer_data.get('transaction_hash')
            )
        except Exception as e:
            print(f"Error getting bridge transfer: {e}")
            return None
    
    @staticmethod
    def shard(shard_id: int) -> Optional[Shard]:
        """Get shard by ID."""
        try:
            shard_data = shard_manager.get_shard(shard_id)
            if not shard_data:
                return None
            
            return Shard(
                shard_id=shard_id,
                validator_count=shard_data.get('validator_count', 0),
                transaction_count=shard_data.get('transaction_count', 0),
                block_height=shard_data.get('block_height', 0),
                status=shard_data.get('status', 'active'),
                validators=shard_data.get('validators', []),
                cross_shard_transactions=shard_data.get('cross_shard_transactions', 0)
            )
        except Exception as e:
            print(f"Error getting shard: {e}")
            return None
    
    @staticmethod
    def validator(address: str) -> Optional[Validator]:
        """Get validator by address."""
        try:
            validator_data = consensus_engine.get_validator(address)
            if not validator_data:
                return None
            
            return Validator(
                address=address,
                public_key=validator_data.get('public_key', ''),
                stake=Decimal(str(validator_data.get('stake', 0))),
                commission_rate=validator_data.get('commission_rate', 0.0),
                status=validator_data.get('status', 'active'),
                uptime=validator_data.get('uptime', 0.0),
                blocks_proposed=validator_data.get('blocks_proposed', 0),
                blocks_validated=validator_data.get('blocks_validated', 0),
                slashing_events=validator_data.get('slashing_events', 0),
                last_active=datetime.fromtimestamp(validator_data.get('last_active', 0))
            )
        except Exception as e:
            print(f"Error getting validator: {e}")
            return None
    
    @staticmethod
    def proposal(proposal_id: str) -> Optional[Proposal]:
        """Get proposal by ID."""
        try:
            proposal_data = governance_engine.get_proposal(proposal_id)
            if not proposal_data:
                return None
            
            return Proposal(
                proposal_id=proposal_id,
                proposer=proposal_data.get('proposer', ''),
                title=proposal_data.get('title', ''),
                description=proposal_data.get('description', ''),
                proposal_type=proposal_data.get('proposal_type', ''),
                status=proposal_data.get('status', 'pending'),
                voting_start=datetime.fromtimestamp(proposal_data.get('voting_start', 0)),
                voting_end=datetime.fromtimestamp(proposal_data.get('voting_end', 0)),
                execution_time=datetime.fromtimestamp(proposal_data.get('execution_time', 0)) if proposal_data.get('execution_time') else None,
                total_votes=proposal_data.get('total_votes', 0),
                yes_votes=proposal_data.get('yes_votes', 0),
                no_votes=proposal_data.get('no_votes', 0),
                abstain_votes=proposal_data.get('abstain_votes', 0),
                quorum=proposal_data.get('quorum', 0.0),
                threshold=proposal_data.get('threshold', 0.0)
            )
        except Exception as e:
            print(f"Error getting proposal: {e}")
            return None
    
    @staticmethod
    def network_stats() -> NetworkStats:
        """Get network statistics."""
        try:
            stats = peer_manager.get_network_stats()
            return NetworkStats(
                total_peers=stats.get('total_peers', 0),
                active_peers=stats.get('active_peers', 0),
                total_connections=stats.get('total_connections', 0),
                bytes_sent=stats.get('bytes_sent', 0),
                bytes_received=stats.get('bytes_received', 0),
                messages_sent=stats.get('messages_sent', 0),
                messages_received=stats.get('messages_received', 0),
                average_latency=stats.get('average_latency', 0.0)
            )
        except Exception as e:
            print(f"Error getting network stats: {e}")
            return NetworkStats(
                total_peers=0, active_peers=0, total_connections=0,
                bytes_sent=0, bytes_received=0, messages_sent=0,
                messages_received=0, average_latency=0.0
            )
    
    @staticmethod
    def performance_metrics() -> PerformanceMetrics:
        """Get performance metrics."""
        try:
            metrics = performance_monitor.get_current_metrics()
            return PerformanceMetrics(
                tps=metrics.get('tps', 0.0),
                latency=metrics.get('latency', 0.0),
                gas_efficiency=metrics.get('gas_efficiency', 0.0),
                memory_usage=metrics.get('memory_usage', 0),
                cpu_usage=metrics.get('cpu_usage', 0.0),
                disk_usage=metrics.get('disk_usage', 0),
                network_throughput=metrics.get('network_throughput', 0.0),
                timestamp=datetime.now()
            )
        except Exception as e:
            print(f"Error getting performance metrics: {e}")
            return PerformanceMetrics(
                tps=0.0, latency=0.0, gas_efficiency=0.0,
                memory_usage=0, cpu_usage=0.0, disk_usage=0,
                network_throughput=0.0, timestamp=datetime.now()
            )

class MutationResolvers:
    """Mutation resolvers implementation."""
    
    @staticmethod
    def create_wallet(name: str, password: Optional[str] = None) -> str:
        """Create a new wallet."""
        try:
            wallet_id = wallet_manager.create_wallet(name, password)
            return wallet_id
        except Exception as e:
            print(f"Error creating wallet: {e}")
            raise Exception(f"Failed to create wallet: {str(e)}")
    
    @staticmethod
    def create_transaction(from_address: str, to_address: str, value: Decimal,
                          gas_price: int, gas_limit: int, data: Optional[str] = None) -> str:
        """Create a new transaction."""
        try:
            tx_data = bytes.fromhex(data) if data else b''
            tx = CoreTransaction(
                sender=from_address,
                recipient=to_address,
                amount=int(value),
                gas_price=gas_price,
                gas_limit=gas_limit,
                data=tx_data
            )
            tx_hash = blockchain.add_transaction(tx)
            return tx_hash
        except Exception as e:
            print(f"Error creating transaction: {e}")
            raise Exception(f"Failed to create transaction: {str(e)}")
    
    @staticmethod
    def deploy_contract(bytecode: str, abi: str, constructor_args: Optional[List[str]] = None) -> str:
        """Deploy a smart contract."""
        try:
            contract = SmartContract(
                bytecode=bytes.fromhex(bytecode),
                abi=json.loads(abi)
            )
            contract_address = blockchain.deploy_contract(contract, constructor_args or [])
            return contract_address
        except Exception as e:
            print(f"Error deploying contract: {e}")
            raise Exception(f"Failed to deploy contract: {str(e)}")
    
    @staticmethod
    def call_contract(contract_address: str, function_name: str, args: Optional[List[str]] = None) -> str:
        """Call a smart contract function."""
        try:
            result = blockchain.call_contract(contract_address, function_name, args or [])
            return result
        except Exception as e:
            print(f"Error calling contract: {e}")
            raise Exception(f"Failed to call contract: {str(e)}")
    
    @staticmethod
    def create_bridge_transfer(source_chain: str, target_chain: str, source_asset: str,
                              target_asset: str, amount: Decimal, receiver: str) -> str:
        """Create a bridge transfer."""
        try:
            transfer_id = bridge_manager.create_transfer(
                source_chain=source_chain,
                target_chain=target_chain,
                source_asset=source_asset,
                target_asset=target_asset,
                amount=int(amount),
                receiver=receiver
            )
            return transfer_id
        except Exception as e:
            print(f"Error creating bridge transfer: {e}")
            raise Exception(f"Failed to create bridge transfer: {str(e)}")
    
    @staticmethod
    def create_proposal(title: str, description: str, proposal_type: str) -> str:
        """Create a governance proposal."""
        try:
            proposal_id = governance_engine.create_proposal(
                title=title,
                description=description,
                proposal_type=proposal_type
            )
            return proposal_id
        except Exception as e:
            print(f"Error creating proposal: {e}")
            raise Exception(f"Failed to create proposal: {str(e)}")
    
    @staticmethod
    def vote(proposal_id: str, vote_option: str, reason: Optional[str] = None) -> bool:
        """Vote on a proposal."""
        try:
            success = governance_engine.vote(proposal_id, vote_option, reason)
            return success
        except Exception as e:
            print(f"Error voting: {e}")
            raise Exception(f"Failed to vote: {str(e)}")

class SubscriptionResolvers:
    """Subscription resolvers implementation."""
    
    @staticmethod
    async def new_blocks() -> AsyncGenerator[Block, None]:
        """Subscribe to new blocks."""
        while True:
            try:
                # Get latest block
                latest_height = blockchain.get_latest_height()
                block = QueryResolvers.block(height=latest_height)
                if block:
                    yield block
                await asyncio.sleep(1)  # Check every second
            except Exception as e:
                print(f"Error in new_blocks subscription: {e}")
                await asyncio.sleep(1)
    
    @staticmethod
    async def new_transactions() -> AsyncGenerator[Transaction, None]:
        """Subscribe to new transactions."""
        while True:
            try:
                # Get pending transactions
                pending_txs = blockchain.get_pending_transactions()
                for tx in pending_txs:
                    yield QueryResolvers.transaction(tx.hash)
                await asyncio.sleep(0.5)  # Check every 500ms
            except Exception as e:
                print(f"Error in new_transactions subscription: {e}")
                await asyncio.sleep(0.5)
    
    @staticmethod
    async def contract_events(contract_address: str) -> AsyncGenerator[ContractEvent, None]:
        """Subscribe to contract events."""
        while True:
            try:
                # Get recent events for contract
                events = blockchain.get_contract_events(contract_address)
                for event in events:
                    yield ContractEvent(
                        contract_address=contract_address,
                        event_name=event.get('event_name', ''),
                        event_data=json.dumps(event.get('event_data', {})),
                        block_height=event.get('block_height', 0),
                        transaction_hash=event.get('transaction_hash', ''),
                        log_index=event.get('log_index', 0),
                        timestamp=datetime.fromtimestamp(event.get('timestamp', 0))
                    )
                await asyncio.sleep(1)
            except Exception as e:
                print(f"Error in contract_events subscription: {e}")
                await asyncio.sleep(1)
    
    @staticmethod
    async def bridge_transfer_updates(transfer_id: str) -> AsyncGenerator[BridgeTransfer, None]:
        """Subscribe to bridge transfer updates."""
        while True:
            try:
                transfer = QueryResolvers.bridge_transfer(transfer_id)
                if transfer:
                    yield transfer
                await asyncio.sleep(2)  # Check every 2 seconds
            except Exception as e:
                print(f"Error in bridge_transfer_updates subscription: {e}")
                await asyncio.sleep(2)
    
    @staticmethod
    async def shard_updates(shard_id: int) -> AsyncGenerator[Shard, None]:
        """Subscribe to shard updates."""
        while True:
            try:
                shard = QueryResolvers.shard(shard_id)
                if shard:
                    yield shard
                await asyncio.sleep(5)  # Check every 5 seconds
            except Exception as e:
                print(f"Error in shard_updates subscription: {e}")
                await asyncio.sleep(5)
    
    @staticmethod
    async def governance_updates() -> AsyncGenerator[Proposal, None]:
        """Subscribe to governance updates."""
        while True:
            try:
                # Get recent proposals
                proposals = governance_engine.get_recent_proposals()
                for proposal_data in proposals:
                    proposal = QueryResolvers.proposal(proposal_data['proposal_id'])
                    if proposal:
                        yield proposal
                await asyncio.sleep(10)  # Check every 10 seconds
            except Exception as e:
                print(f"Error in governance_updates subscription: {e}")
                await asyncio.sleep(10)
    
    @staticmethod
    async def network_stats_updates() -> AsyncGenerator[NetworkStats, None]:
        """Subscribe to network statistics updates."""
        while True:
            try:
                yield QueryResolvers.network_stats()
                await asyncio.sleep(5)  # Check every 5 seconds
            except Exception as e:
                print(f"Error in network_stats_updates subscription: {e}")
                await asyncio.sleep(5)
    
    @staticmethod
    async def performance_metrics_updates() -> AsyncGenerator[PerformanceMetrics, None]:
        """Subscribe to performance metrics updates."""
        while True:
            try:
                yield QueryResolvers.performance_metrics()
                await asyncio.sleep(1)  # Check every second
            except Exception as e:
                print(f"Error in performance_metrics_updates subscription: {e}")
                await asyncio.sleep(1)
