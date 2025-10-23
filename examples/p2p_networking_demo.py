#!/usr/bin/env python3
logger = logging.getLogger(__name__)
"""
Advanced P2P Networking Demo for GodChain.

This demo showcases the sophisticated P2P networking system including
gossip protocols, peer discovery, connection management, and network optimization.
"""

import logging
import asyncio
import time
import random
from dubchain.network import (
    Peer, PeerInfo, ConnectionType, PeerStatus,
    GossipProtocol, GossipMessage, MessageType, GossipConfig,
    PeerDiscovery, DiscoveryMethod, DiscoveryConfig,
    ConnectionManager, ConnectionStrategy, ConnectionConfig
)
from dubchain.crypto.signatures import PrivateKey, PublicKey


class P2PNetworkDemo:
    """P2P Network demonstration."""
    
    def __init__(self, node_id: str, port: int):
        """Initialize demo."""
        self.node_id = node_id
        self.port = port
        self.private_key = PrivateKey.generate()
        self.public_key = self.private_key.get_public_key()
        
        # Initialize components
        self.gossip_config = GossipConfig(
            fanout=3,
            interval=2.0,
            max_messages=1000,
            message_ttl=300,
            max_hops=5
        )
        self.gossip_protocol = GossipProtocol(self.gossip_config, node_id)
        
        self.discovery_config = DiscoveryConfig(
            discovery_interval=10.0,
            max_peers=20,
            min_peers=3,
            peer_timeout=5.0,
            enable_peer_exchange=True
        )
        self.peer_discovery = PeerDiscovery(self.discovery_config, node_id, self.private_key)
        
        self.connection_config = ConnectionConfig(
            max_connections=15,
            min_connections=2,
            connection_timeout=5.0,
            connection_strategy=ConnectionStrategy.LOAD_BALANCED
        )
        self.connection_manager = ConnectionManager(self.connection_config, node_id, self.private_key)
        
        # Demo state
        self.messages_received = []
        self.peers_connected = []
        self.network_events = []
    
    async def start(self) -> None:
        """Start the P2P network demo."""
        logger.info(f"🚀 Starting P2P Network Demo - Node: {self.node_id}")
        logger.info(f"   Port: {self.port}")
        logger.info(f"   Public Key: {self.public_key.to_hex()[:16]}...")
        
        # Start components
        await self.gossip_protocol.start()
        await self.peer_discovery.start()
        await self.connection_manager.start()
        
        # Add callbacks
        self.peer_discovery.add_discovery_callback(self._on_peer_discovered)
        self.connection_manager.add_connection_callback(self._on_peer_connected)
        self.connection_manager.add_disconnection_callback(self._on_peer_disconnected)
        
        # Add message handlers
        self.gossip_protocol.add_message_handler(MessageType.CUSTOM, self._handle_custom_message)
        self.gossip_protocol.add_message_handler(MessageType.ANNOUNCEMENT, self._handle_announcement)
        
        logger.info("✅ P2P Network components started successfully!")
    
    async def stop(self) -> None:
        """Stop the P2P network demo."""
        logger.info(f"🛑 Stopping P2P Network Demo - Node: {self.node_id}")
        
        await self.gossip_protocol.stop()
        await self.peer_discovery.stop()
        await self.connection_manager.stop()
        
        logger.info("✅ P2P Network components stopped successfully!")
    
    async def add_bootstrap_peer(self, address: str, port: int) -> None:
        """Add bootstrap peer."""
        peer_info = PeerInfo(
            peer_id=f"bootstrap_{address}_{port}",
            public_key=PrivateKey.generate().get_public_key(),  # Placeholder
            address=address,
            port=port,
            connection_type=ConnectionType.SEED
        )
        
        await self.peer_discovery.add_peer(peer_info)
        logger.info(f"📡 Added bootstrap peer: {address}:{port}")
    
    async def discover_peers(self) -> None:
        """Discover peers in the network."""
        logger.info("🔍 Discovering peers...")
        
        # Try different discovery methods
        methods = [DiscoveryMethod.BOOTSTRAP, DiscoveryMethod.PEER_EXCHANGE]
        
        for method in methods:
            try:
                peers = await self.peer_discovery.discover_peers(method)
                logger.info(f"   {method.value}: Found {len(peers)} peers")
                
                for peer_info in peers:
                    await self.peer_discovery.add_peer(peer_info)
                    
            except Exception as e:
                logger.info(f"   {method.value}: Discovery failed - {e}")
    
    async def connect_to_peers(self) -> None:
        """Connect to discovered peers."""
        logger.info("🔗 Connecting to peers...")
        
        discovered_peers = await self.peer_discovery.get_peers()
        logger.info(f"   Found {len(discovered_peers)} discovered peers")
        
        connected_count = 0
        for peer_info in discovered_peers:
            if await self.connection_manager.connection_pool.can_add_connection():
                peer = await self.connection_manager.connect_to_peer(peer_info)
                if peer:
                    connected_count += 1
                    logger.info(f"   ✅ Connected to {peer_info.peer_id} at {peer_info.address}:{peer_info.port}")
                else:
                    logger.info(f"   ❌ Failed to connect to {peer_info.peer_id}")
            else:
                logger.info(f"   ⚠️  Max connections reached, skipping {peer_info.peer_id}")
        
        logger.info(f"   Connected to {connected_count} peers")
    
    async def broadcast_message(self, message_content: str) -> None:
        """Broadcast a message using gossip protocol."""
        logger.info(f"📢 Broadcasting message: {message_content}")
        
        message_id = await self.gossip_protocol.broadcast_message(
            MessageType.CUSTOM,
            {
                'content': message_content,
                'sender': self.node_id,
                'timestamp': int(time.time())
            }
        )
        
        logger.info(f"   Message ID: {message_id}")
        return message_id
    
    async def send_announcement(self, announcement: str) -> None:
        """Send an announcement."""
        logger.info(f"📢 Sending announcement: {announcement}")
        
        await self.gossip_protocol.broadcast_message(
            MessageType.ANNOUNCEMENT,
            {
                'announcement': announcement,
                'node_id': self.node_id,
                'timestamp': int(time.time())
            }
        )
    
    async def ping_peers(self) -> None:
        """Ping all connected peers."""
        logger.info("🏓 Pinging peers...")
        
        connected_peers = await self.connection_manager.connection_pool.get_available_connections()
        
        for peer in connected_peers:
            try:
                latency = await peer.ping()
                if latency is not None:
                    logger.info(f"   {peer.get_peer_id()}: {latency:.2f}ms")
                else:
                    logger.info(f"   {peer.get_peer_id()}: No response")
            except Exception as e:
                logger.info(f"   {peer.get_peer_id()}: Error - {e}")
    
    async def show_network_stats(self) -> None:
        """Show network statistics."""
        logger.info("📊 Network Statistics:")
        
        # Gossip stats
        gossip_stats = self.gossip_protocol.get_stats()
        logger.info(f"   Gossip Protocol:")
        logger.info(f"     - Peers: {gossip_stats['peers_count']}")
        logger.info(f"     - Messages: {gossip_stats['messages_count']}")
        logger.info(f"     - Running: {gossip_stats['running']}")
        
        # Discovery stats
        discovery_stats = self.peer_discovery.get_stats()
        logger.info(f"   Peer Discovery:")
        logger.info(f"     - Discovered: {discovery_stats['discovered_peers_count']}")
        logger.info(f"     - Connected: {discovery_stats['connected_peers_count']}")
        logger.info(f"     - Bootstrap: {discovery_stats['bootstrap_peers_count']}")
        
        # Connection stats
        connection_stats = self.connection_manager.get_stats()
        logger.info(f"   Connection Manager:")
        logger.info(f"     - Connections: {connection_stats['connections_count']}")
        logger.info(f"     - Queue: {connection_stats['connection_queue_count']}")
        logger.info(f"     - Failed: {connection_stats['failed_connections_count']}")
        
        # Message stats
        logger.info(f"   Messages:")
        logger.info(f"     - Received: {len(self.messages_received)}")
        logger.info(f"     - Network Events: {len(self.network_events)}")
    
    async def _on_peer_discovered(self, peer_info: PeerInfo) -> None:
        """Handle peer discovery."""
        event = {
            'type': 'peer_discovered',
            'peer_id': peer_info.peer_id,
            'address': f"{peer_info.address}:{peer_info.port}",
            'timestamp': int(time.time())
        }
        self.network_events.append(event)
        logger.info(f"🔍 Discovered peer: {peer_info.peer_id} at {peer_info.address}:{peer_info.port}")
    
    async def _on_peer_connected(self, peer: Peer) -> None:
        """Handle peer connection."""
        self.peers_connected.append(peer.get_peer_id())
        event = {
            'type': 'peer_connected',
            'peer_id': peer.get_peer_id(),
            'address': peer.get_address(),
            'timestamp': int(time.time())
        }
        self.network_events.append(event)
        logger.info(f"🔗 Connected to peer: {peer.get_peer_id()}")
    
    async def _on_peer_disconnected(self, peer: Peer) -> None:
        """Handle peer disconnection."""
        if peer.get_peer_id() in self.peers_connected:
            self.peers_connected.remove(peer.get_peer_id())
        
        event = {
            'type': 'peer_disconnected',
            'peer_id': peer.get_peer_id(),
            'address': peer.get_address(),
            'timestamp': int(time.time())
        }
        self.network_events.append(event)
        logger.info(f"🔌 Disconnected from peer: {peer.get_peer_id()}")
    
    async def _handle_custom_message(self, peer: Peer, message: GossipMessage) -> None:
        """Handle custom message."""
        content = message.content
        if isinstance(content, dict) and 'content' in content:
            message_text = content['content']
            sender = content.get('sender', 'unknown')
            
            self.messages_received.append({
                'type': 'custom_message',
                'sender': sender,
                'content': message_text,
                'peer_id': peer.get_peer_id(),
                'timestamp': int(time.time())
            })
            
            logger.info(f"📨 Received message from {sender}: {message_text}")
    
    async def _handle_announcement(self, peer: Peer, message: GossipMessage) -> None:
        """Handle announcement."""
        content = message.content
        if isinstance(content, dict) and 'announcement' in content:
            announcement = content['announcement']
            sender = content.get('node_id', 'unknown')
            
            self.messages_received.append({
                'type': 'announcement',
                'sender': sender,
                'announcement': announcement,
                'peer_id': peer.get_peer_id(),
                'timestamp': int(time.time())
            })
            
            logger.info(f"📢 Received announcement from {sender}: {announcement}")


async def run_network_demo():
    """Run the P2P network demonstration."""
    logger.info("🌐 GodChain Advanced P2P Networking Demo")
    logger.info("=" * 60)
    
    # Create demo nodes
    nodes = []
    base_port = 8000
    
    for i in range(3):
        node_id = f"node_{i+1}"
        port = base_port + i
        node = P2PNetworkDemo(node_id, port)
        nodes.append(node)
    
    try:
        # Start all nodes
        logger.info("\n🚀 Starting network nodes...")
        for node in nodes:
            await node.start()
            await asyncio.sleep(0.5)  # Small delay between starts
        
        # Add bootstrap peers (each node knows about the others)
        logger.info("\n📡 Setting up bootstrap peers...")
        for i, node in enumerate(nodes):
            for j, other_node in enumerate(nodes):
                if i != j:
                    await node.add_bootstrap_peer("127.0.0.1", other_node.port)
        
        # Wait for network to stabilize
        logger.info("\n⏳ Waiting for network to stabilize...")
        await asyncio.sleep(3)
        
        # Discover peers
        logger.info("\n🔍 Discovering peers...")
        for node in nodes:
            await node.discover_peers()
            await asyncio.sleep(1)
        
        # Connect to peers
        logger.info("\n🔗 Connecting to peers...")
        for node in nodes:
            await node.connect_to_peers()
            await asyncio.sleep(1)
        
        # Wait for connections to establish
        logger.info("\n⏳ Waiting for connections to establish...")
        await asyncio.sleep(2)
        
        # Show initial stats
        logger.info("\n📊 Initial Network Statistics:")
        for node in nodes:
            logger.info(f"\n--- {node.node_id} ---")
            await node.show_network_stats()
        
        # Demonstrate messaging
        logger.info("\n💬 Demonstrating messaging...")
        
        # Node 1 sends a message
        await nodes[0].broadcast_message("Hello from Node 1! 🌟")
        await asyncio.sleep(1)
        
        # Node 2 sends an announcement
        await nodes[1].send_announcement("Node 2 is online and ready! 🚀")
        await asyncio.sleep(1)
        
        # Node 3 sends a message
        await nodes[2].broadcast_message("Node 3 reporting in! 💪")
        await asyncio.sleep(1)
        
        # Ping peers
        logger.info("\n🏓 Pinging peers...")
        for node in nodes:
            logger.info(f"\n--- {node.node_id} ---")
            await node.ping_peers()
        
        # Show final stats
        logger.info("\n📊 Final Network Statistics:")
        for node in nodes:
            logger.info(f"\n--- {node.node_id} ---")
            await node.show_network_stats()
        
        # Demonstrate network resilience
        logger.info("\n🛡️ Demonstrating network resilience...")
        
        # Simulate node disconnection
        logger.info("   Simulating node disconnection...")
        await nodes[1].stop()
        await asyncio.sleep(2)
        
        # Show stats after disconnection
        logger.info("   Network stats after disconnection:")
        for i, node in enumerate(nodes):
            if i != 1:  # Skip the stopped node
                logger.info(f"\n--- {node.node_id} ---")
                await node.show_network_stats()
        
        # Restart the node
        logger.info("   Restarting disconnected node...")
        await nodes[1].start()
        await asyncio.sleep(2)
        
        # Reconnect
        await nodes[1].discover_peers()
        await nodes[1].connect_to_peers()
        await asyncio.sleep(2)
        
        # Final message
        await nodes[1].broadcast_message("Node 2 is back online! 🔄")
        
        logger.info("\n🎉 P2P Network Demo Completed Successfully!")
        logger.info("=" * 60)
        logger.info("✨ Features demonstrated:")
        logger.info("   - Advanced peer discovery")
        logger.info("   - Gossip protocol messaging")
        logger.info("   - Connection management")
        logger.info("   - Network resilience")
        logger.info("   - Real-time communication")
        logger.info("   - Automatic reconnection")
        logger.info("   - Load balancing")
        logger.info("   - Network monitoring")
        
    finally:
        # Stop all nodes
        logger.info("\n🛑 Stopping all nodes...")
        for node in nodes:
            await node.stop()


async def main():
    """Main demo function."""
    await run_network_demo()


if __name__ == "__main__":
    asyncio.run(main())
