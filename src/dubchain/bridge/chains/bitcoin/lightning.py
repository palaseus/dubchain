"""
Lightning Network Integration for Bitcoin

This module provides comprehensive Lightning Network integration including:
- Channel management and operations
- Payment routing and forwarding
- Invoice generation and payment
- Network topology management
- Lightning Network bridge functionality
"""

import asyncio
import hashlib
import json
import time
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from decimal import Decimal
import secrets
import base64

try:
    import lightning
    from lightning import LightningRpc
    LIGHTNING_AVAILABLE = True
except ImportError:
    LIGHTNING_AVAILABLE = False

from ....errors import BridgeError, ClientError
from ....logging import get_logger
from .client import BitcoinClient, BitcoinConfig
from .bridge import BitcoinBridge, BridgeConfig

logger = get_logger(__name__)


@dataclass
class LightningConfig:
    """Configuration for Lightning Network."""
    rpc_path: str = "/tmp/lightning-rpc"
    network: str = "bitcoin"  # bitcoin, testnet, regtest
    enable_autopilot: bool = True
    max_channels: int = 10
    min_channel_size: int = 100000  # satoshis
    max_channel_size: int = 10000000  # satoshis
    fee_rate: int = 1000  # satoshis per 1000 blocks
    base_fee: int = 1000  # satoshis
    cltv_delta: int = 144  # blocks
    enable_htlc_interceptor: bool = True


@dataclass
class LightningChannel:
    """Lightning Network channel data."""
    channel_id: str
    peer_id: str
    local_balance: int
    remote_balance: int
    capacity: int
    state: str  # OPENING, CHANNELD_NORMAL, CLOSING, CLOSED
    is_active: bool
    is_public: bool
    created_at: float = field(default_factory=time.time)


@dataclass
class LightningInvoice:
    """Lightning Network invoice."""
    payment_hash: str
    payment_request: str
    amount: int
    description: str
    expiry: int
    created_at: float = field(default_factory=time.time)
    paid: bool = False
    paid_at: Optional[float] = None


@dataclass
class LightningPayment:
    """Lightning Network payment."""
    payment_hash: str
    payment_preimage: str
    amount: int
    fee: int
    destination: str
    status: str  # pending, succeeded, failed
    created_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None


@dataclass
class LightningNode:
    """Lightning Network node information."""
    node_id: str
    alias: str
    color: str
    features: Dict[str, Any]
    addresses: List[Dict[str, Any]]
    last_update: float


class LightningManager:
    """Manages Lightning Network operations."""
    
    def __init__(self, config: LightningConfig):
        self.config = config
        self.lightning_rpc: Optional[LightningRpc] = None
        self._initialized = False
        
        if LIGHTNING_AVAILABLE:
            self._initialize_lightning()
    
    def _initialize_lightning(self) -> None:
        """Initialize Lightning Network connection."""
        try:
            self.lightning_rpc = LightningRpc(self.config.rpc_path)
            self._initialized = True
            logger.info("Lightning Network connection initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Lightning Network: {e}")
            self._initialized = False
    
    def is_connected(self) -> bool:
        """Check if Lightning Network is connected."""
        return self._initialized and self.lightning_rpc is not None
    
    async def get_info(self) -> Dict[str, Any]:
        """Get Lightning Network node information."""
        if not self.is_connected():
            raise RuntimeError("Lightning Network not connected")
        
        try:
            info = self.lightning_rpc.getinfo()
            return info
        except Exception as e:
            logger.error(f"Failed to get Lightning info: {e}")
            raise BridgeError(f"Failed to get Lightning info: {e}")
    
    async def list_channels(self) -> List[LightningChannel]:
        """List all Lightning Network channels."""
        if not self.is_connected():
            raise RuntimeError("Lightning Network not connected")
        
        try:
            channels_data = self.lightning_rpc.listchannels()
            channels = []
            
            for channel_data in channels_data:
                channel = LightningChannel(
                    channel_id=channel_data["channel_id"],
                    peer_id=channel_data["source"],
                    local_balance=channel_data.get("local_balance", 0),
                    remote_balance=channel_data.get("remote_balance", 0),
                    capacity=channel_data["satoshis"],
                    state=channel_data.get("state", "UNKNOWN"),
                    is_active=channel_data.get("active", False),
                    is_public=channel_data.get("public", False)
                )
                channels.append(channel)
            
            return channels
            
        except Exception as e:
            logger.error(f"Failed to list channels: {e}")
            raise BridgeError(f"Failed to list channels: {e}")
    
    async def create_channel(self, peer_id: str, amount: int, 
                           public: bool = True) -> str:
        """Create a new Lightning Network channel."""
        if not self.is_connected():
            raise RuntimeError("Lightning Network not connected")
        
        try:
            result = self.lightning_rpc.fundchannel(
                peer_id, 
                amount,
                announce=public
            )
            return result["txid"]
            
        except Exception as e:
            logger.error(f"Failed to create channel: {e}")
            raise BridgeError(f"Failed to create channel: {e}")
    
    async def close_channel(self, channel_id: str, force: bool = False) -> str:
        """Close a Lightning Network channel."""
        if not self.is_connected():
            raise RuntimeError("Lightning Network not connected")
        
        try:
            if force:
                result = self.lightning_rpc.close(channel_id, force=True)
            else:
                result = self.lightning_rpc.close(channel_id)
            
            return result["txid"]
            
        except Exception as e:
            logger.error(f"Failed to close channel: {e}")
            raise BridgeError(f"Failed to close channel: {e}")
    
    async def create_invoice(self, amount: int, description: str = "", 
                           expiry: int = 3600) -> LightningInvoice:
        """Create a Lightning Network invoice."""
        if not self.is_connected():
            raise RuntimeError("Lightning Network not connected")
        
        try:
            result = self.lightning_rpc.invoice(
                amount,
                f"invoice_{secrets.token_hex(8)}",
                description,
                expiry
            )
            
            invoice = LightningInvoice(
                payment_hash=result["payment_hash"],
                payment_request=result["bolt11"],
                amount=amount,
                description=description,
                expiry=expiry
            )
            
            return invoice
            
        except Exception as e:
            logger.error(f"Failed to create invoice: {e}")
            raise BridgeError(f"Failed to create invoice: {e}")
    
    async def pay_invoice(self, payment_request: str, 
                        max_fee: Optional[int] = None) -> LightningPayment:
        """Pay a Lightning Network invoice."""
        if not self.is_connected():
            raise RuntimeError("Lightning Network not connected")
        
        try:
            # Decode invoice to get amount
            decoded = self.lightning_rpc.decodepay(payment_request)
            amount = decoded["amount_msat"] // 1000  # Convert to satoshis
            
            # Pay invoice
            result = self.lightning_rpc.pay(
                payment_request,
                maxfeepercent=0.1,  # 0.1% max fee
                maxfee=max_fee or (amount * 0.01)  # 1% of amount as max fee
            )
            
            payment = LightningPayment(
                payment_hash=result["payment_hash"],
                payment_preimage=result["payment_preimage"],
                amount=amount,
                fee=result["fee_msat"] // 1000,
                destination=decoded["payee"],
                status="succeeded",
                completed_at=time.time()
            )
            
            return payment
            
        except Exception as e:
            logger.error(f"Failed to pay invoice: {e}")
            raise BridgeError(f"Failed to pay invoice: {e}")
    
    async def get_route(self, destination: str, amount: int, 
                       max_fee: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get route for payment."""
        if not self.is_connected():
            raise RuntimeError("Lightning Network not connected")
        
        try:
            route = self.lightning_rpc.getroute(
                destination,
                amount,
                riskfactor=10,
                maxfeepercent=0.1,
                maxfee=max_fee or (amount * 0.01)
            )
            
            return route
            
        except Exception as e:
            logger.error(f"Failed to get route: {e}")
            raise BridgeError(f"Failed to get route: {e}")
    
    async def list_peers(self) -> List[Dict[str, Any]]:
        """List Lightning Network peers."""
        if not self.is_connected():
            raise RuntimeError("Lightning Network not connected")
        
        try:
            peers = self.lightning_rpc.listpeers()
            return peers
            
        except Exception as e:
            logger.error(f"Failed to list peers: {e}")
            raise BridgeError(f"Failed to list peers: {e}")
    
    async def connect_peer(self, peer_id: str, host: str, port: int) -> bool:
        """Connect to a Lightning Network peer."""
        if not self.is_connected():
            raise RuntimeError("Lightning Network not connected")
        
        try:
            result = self.lightning_rpc.connect(peer_id, f"{host}:{port}")
            return result["id"] == peer_id
            
        except Exception as e:
            logger.error(f"Failed to connect peer: {e}")
            raise BridgeError(f"Failed to connect peer: {e}")
    
    async def get_node_info(self, node_id: str) -> LightningNode:
        """Get Lightning Network node information."""
        if not self.is_connected():
            raise RuntimeError("Lightning Network not connected")
        
        try:
            info = self.lightning_rpc.listnodes(node_id)
            
            if not info:
                raise BridgeError(f"Node {node_id} not found")
            
            node_data = info[0]
            node = LightningNode(
                node_id=node_data["nodeid"],
                alias=node_data.get("alias", ""),
                color=node_data.get("color", ""),
                features=node_data.get("features", {}),
                addresses=node_data.get("addresses", []),
                last_update=node_data.get("last_timestamp", 0)
            )
            
            return node
            
        except Exception as e:
            logger.error(f"Failed to get node info: {e}")
            raise BridgeError(f"Failed to get node info: {e}")


class LightningBridge:
    """Lightning Network bridge for Bitcoin."""
    
    def __init__(self, lightning_config: LightningConfig, 
                 bridge_config: BridgeConfig):
        self.lightning_config = lightning_config
        self.bridge_config = bridge_config
        self.lightning_manager = LightningManager(lightning_config)
        self.bitcoin_bridge = BitcoinBridge(bridge_config)
        self._running = False
        
    async def start(self) -> None:
        """Start the Lightning Network bridge."""
        if self._running:
            return
            
        self._running = True
        await self.bitcoin_bridge.start()
        logger.info("Lightning Network bridge started")
        
    async def stop(self) -> None:
        """Stop the Lightning Network bridge."""
        self._running = False
        await self.bitcoin_bridge.stop()
        logger.info("Lightning Network bridge stopped")
    
    async def get_lightning_balance(self) -> Dict[str, int]:
        """Get Lightning Network balance."""
        try:
            channels = await self.lightning_manager.list_channels()
            
            total_local = sum(channel.local_balance for channel in channels)
            total_remote = sum(channel.remote_balance for channel in channels)
            total_capacity = sum(channel.capacity for channel in channels)
            
            return {
                "local_balance": total_local,
                "remote_balance": total_remote,
                "total_capacity": total_capacity,
                "available_balance": total_local,
                "channel_count": len(channels)
            }
            
        except Exception as e:
            logger.error(f"Failed to get Lightning balance: {e}")
            raise BridgeError(f"Failed to get Lightning balance: {e}")
    
    async def create_lightning_invoice(self, amount: int, description: str = "") -> LightningInvoice:
        """Create a Lightning Network invoice."""
        return await self.lightning_manager.create_invoice(amount, description)
    
    async def pay_lightning_invoice(self, payment_request: str) -> LightningPayment:
        """Pay a Lightning Network invoice."""
        return await self.lightning_manager.pay_invoice(payment_request)
    
    async def bridge_to_lightning(self, bitcoin_address: str, amount: int) -> str:
        """Bridge Bitcoin to Lightning Network."""
        try:
            # Get Lightning Network address
            lightning_info = await self.lightning_manager.get_info()
            lightning_address = lightning_info["id"]
            
            # Send Bitcoin to Lightning Network
            tx_id = await self.bitcoin_bridge.send_transaction(
                bitcoin_address, 
                lightning_address, 
                amount
            )
            
            return tx_id
            
        except Exception as e:
            logger.error(f"Failed to bridge to Lightning: {e}")
            raise BridgeError(f"Failed to bridge to Lightning: {e}")
    
    async def bridge_from_lightning(self, lightning_address: str, amount: int) -> str:
        """Bridge from Lightning Network to Bitcoin."""
        try:
            # Create Lightning invoice
            invoice = await self.lightning_manager.create_invoice(amount)
            
            # Pay invoice (this would be done by the recipient)
            payment = await self.lightning_manager.pay_invoice(invoice.payment_request)
            
            return payment.payment_hash
            
        except Exception as e:
            logger.error(f"Failed to bridge from Lightning: {e}")
            raise BridgeError(f"Failed to bridge from Lightning: {e}")
    
    async def get_lightning_stats(self) -> Dict[str, Any]:
        """Get Lightning Network statistics."""
        try:
            info = await self.lightning_manager.get_info()
            channels = await self.lightning_manager.list_channels()
            balance = await self.get_lightning_balance()
            
            return {
                "node_id": info["id"],
                "alias": info.get("alias", ""),
                "network": info.get("network", ""),
                "block_height": info.get("blockheight", 0),
                "channel_count": len(channels),
                "total_capacity": balance["total_capacity"],
                "local_balance": balance["local_balance"],
                "remote_balance": balance["remote_balance"],
                "lightning_available": self.lightning_manager.is_connected(),
                "running": self._running
            }
            
        except Exception as e:
            logger.error(f"Failed to get Lightning stats: {e}")
            return {"error": str(e), "running": self._running}
    
    async def optimize_channels(self) -> Dict[str, Any]:
        """Optimize Lightning Network channels."""
        try:
            channels = await self.lightning_manager.list_channels()
            
            # Find channels that need rebalancing
            unbalanced_channels = []
            for channel in channels:
                if channel.local_balance < channel.capacity * 0.1:  # Less than 10% local balance
                    unbalanced_channels.append(channel)
            
            # Rebalance channels (simplified implementation)
            rebalance_results = []
            for channel in unbalanced_channels:
                try:
                    # This would involve complex rebalancing logic
                    # For now, just log the channel
                    rebalance_results.append({
                        "channel_id": channel.channel_id,
                        "action": "rebalance_needed",
                        "local_balance": channel.local_balance,
                        "capacity": channel.capacity
                    })
                except Exception as e:
                    logger.error(f"Failed to rebalance channel {channel.channel_id}: {e}")
            
            return {
                "total_channels": len(channels),
                "unbalanced_channels": len(unbalanced_channels),
                "rebalance_results": rebalance_results
            }
            
        except Exception as e:
            logger.error(f"Failed to optimize channels: {e}")
            raise BridgeError(f"Failed to optimize channels: {e}")
    
    async def get_network_topology(self) -> Dict[str, Any]:
        """Get Lightning Network topology."""
        try:
            peers = await self.lightning_manager.list_peers()
            channels = await self.lightning_manager.list_channels()
            
            # Build topology graph
            nodes = {}
            edges = []
            
            for peer in peers:
                nodes[peer["id"]] = {
                    "id": peer["id"],
                    "connected": peer.get("connected", False),
                    "channels": []
                }
            
            for channel in channels:
                if channel.peer_id in nodes:
                    nodes[channel.peer_id]["channels"].append({
                        "channel_id": channel.channel_id,
                        "capacity": channel.capacity,
                        "local_balance": channel.local_balance,
                        "remote_balance": channel.remote_balance,
                        "is_active": channel.is_active
                    })
                
                edges.append({
                    "source": channel.peer_id,
                    "target": "local_node",
                    "channel_id": channel.channel_id,
                    "capacity": channel.capacity
                })
            
            return {
                "nodes": list(nodes.values()),
                "edges": edges,
                "total_nodes": len(nodes),
                "total_channels": len(channels)
            }
            
        except Exception as e:
            logger.error(f"Failed to get network topology: {e}")
            raise BridgeError(f"Failed to get network topology: {e}")
