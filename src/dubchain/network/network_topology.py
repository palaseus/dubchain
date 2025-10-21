"""
Network Topology Module

This module provides network topology management including:
- Dynamic topology discovery and maintenance
- Topology optimization algorithms
- Network partitioning detection and recovery
- Load balancing and traffic distribution
- Topology-aware routing and propagation
"""

import asyncio
import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import uuid
from collections import defaultdict, deque

from ..errors import NetworkError, ValidationError
from ..logging import get_logger

logger = get_logger(__name__)

class TopologyType(Enum):
    """Types of network topologies."""
    MESH = "mesh"
    STAR = "star"
    RING = "ring"
    TREE = "tree"
    HYBRID = "hybrid"
    ADAPTIVE = "adaptive"

class NodeRole(Enum):
    """Node roles in the topology."""
    CORE = "core"
    EDGE = "edge"
    BRIDGE = "bridge"
    LEAF = "leaf"
    HUB = "hub"

class ConnectionQuality(Enum):
    """Connection quality levels."""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    UNREACHABLE = "unreachable"

@dataclass
class NetworkTopology:
    """Network topology representation."""
    topology_id: str
    topology_type: TopologyType
    nodes: Dict[str, 'NetworkNode']
    links: Dict[str, 'NetworkLink']
    created_at: float = field(default_factory=time.time)
    last_updated: float = field(default_factory=time.time)
    metrics: Optional['TopologyMetrics'] = None
    
    def get_node(self, node_id: str) -> Optional['NetworkNode']:
        """Get node by ID."""
        return self.nodes.get(node_id)
    
    def get_link(self, link_id: str) -> Optional['NetworkLink']:
        """Get link by ID."""
        return self.links.get(link_id)
    
    def add_node(self, node: 'NetworkNode') -> None:
        """Add node to topology."""
        self.nodes[node.node_id] = node
        self.last_updated = time.time()
    
    def remove_node(self, node_id: str) -> None:
        """Remove node from topology."""
        if node_id in self.nodes:
            del self.nodes[node_id]
            # Remove all links connected to this node
            links_to_remove = [
                link_id for link_id, link in self.links.items()
                if link.source_node_id == node_id or link.target_node_id == node_id
            ]
            for link_id in links_to_remove:
                del self.links[link_id]
            self.last_updated = time.time()
    
    def add_link(self, link: 'NetworkLink') -> None:
        """Add link to topology."""
        self.links[link.link_id] = link
        self.last_updated = time.time()
    
    def remove_link(self, link_id: str) -> None:
        """Remove link from topology."""
        if link_id in self.links:
            del self.links[link_id]
            self.last_updated = time.time()
    
    def get_neighbors(self, node_id: str) -> List[str]:
        """Get neighbor nodes for a given node."""
        neighbors = []
        for link in self.links.values():
            if link.source_node_id == node_id:
                neighbors.append(link.target_node_id)
            elif link.target_node_id == node_id:
                neighbors.append(link.source_node_id)
        return neighbors
    
    def is_connected(self) -> bool:
        """Check if topology is connected."""
        if not self.nodes:
            return False
        
        # Use DFS to check connectivity
        visited = set()
        start_node = next(iter(self.nodes.keys()))
        stack = [start_node]
        
        while stack:
            node_id = stack.pop()
            if node_id not in visited:
                visited.add(node_id)
                neighbors = self.get_neighbors(node_id)
                stack.extend(neighbors)
        
        return len(visited) == len(self.nodes)
    
    def get_shortest_path(self, source: str, target: str) -> Optional[List[str]]:
        """Get shortest path between two nodes."""
        if source not in self.nodes or target not in self.nodes:
            return None
        
        if source == target:
            return [source]
        
        # Use BFS to find shortest path
        queue = [(source, [source])]
        visited = {source}
        
        while queue:
            current, path = queue.pop(0)
            neighbors = self.get_neighbors(current)
            
            for neighbor in neighbors:
                if neighbor == target:
                    return path + [neighbor]
                
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        
        return None
    
    def __str__(self) -> str:
        """String representation."""
        return f"NetworkTopology(id={self.topology_id}, type={self.topology_type.value}, nodes={len(self.nodes)}, links={len(self.links)})"
    
    def __repr__(self) -> str:
        """Representation."""
        return self.__str__()

@dataclass
class NetworkNode:
    """Represents a node in the network topology."""
    node_id: str
    address: str
    port: int
    role: NodeRole
    capabilities: Set[str] = field(default_factory=set)
    metrics: Dict[str, float] = field(default_factory=dict)
    last_seen: float = field(default_factory=time.time)
    is_active: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class NetworkLink:
    """Represents a link between two nodes."""
    source_id: str
    target_id: str
    quality: ConnectionQuality
    latency: float = 0.0
    bandwidth: float = 0.0
    reliability: float = 1.0
    last_updated: float = field(default_factory=time.time)
    is_active: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TopologyMetrics:
    """Metrics for network topology."""
    total_nodes: int = 0
    active_nodes: int = 0
    total_links: int = 0
    active_links: int = 0
    average_latency: float = 0.0
    average_bandwidth: float = 0.0
    network_diameter: int = 0
    clustering_coefficient: float = 0.0
    connectivity_ratio: float = 0.0

class TopologyDiscovery:
    """Discovers and maintains network topology."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize topology discovery."""
        self.config = config
        self.nodes: Dict[str, NetworkNode] = {}
        self.links: Dict[Tuple[str, str], NetworkLink] = {}
        self.discovery_interval = config.get("discovery_interval", 30)
        self.link_timeout = config.get("link_timeout", 60)
        logger.info("Initialized topology discovery")
    
    def add_node(self, node: NetworkNode) -> None:
        """Add a node to the topology."""
        try:
            self.nodes[node.node_id] = node
            logger.debug(f"Added node {node.node_id} to topology")
            
        except Exception as e:
            logger.error(f"Error adding node {node.node_id}: {e}")
    
    def remove_node(self, node_id: str) -> None:
        """Remove a node from the topology."""
        try:
            if node_id in self.nodes:
                del self.nodes[node_id]
                
                # Remove all links involving this node
                links_to_remove = [
                    link_key for link_key in self.links.keys()
                    if node_id in link_key
                ]
                
                for link_key in links_to_remove:
                    del self.links[link_key]
                
                logger.debug(f"Removed node {node_id} and its links from topology")
            
        except Exception as e:
            logger.error(f"Error removing node {node_id}: {e}")
    
    def add_link(self, link: NetworkLink) -> None:
        """Add a link to the topology."""
        try:
            link_key = (link.source_id, link.target_id)
            self.links[link_key] = link
            logger.debug(f"Added link {link.source_id} -> {link.target_id}")
            
        except Exception as e:
            logger.error(f"Error adding link: {e}")
    
    def remove_link(self, source_id: str, target_id: str) -> None:
        """Remove a link from the topology."""
        try:
            link_key = (source_id, target_id)
            if link_key in self.links:
                del self.links[link_key]
                logger.debug(f"Removed link {source_id} -> {target_id}")
            
        except Exception as e:
            logger.error(f"Error removing link: {e}")
    
    def update_link_quality(self, source_id: str, target_id: str, 
                          latency: float, bandwidth: float, reliability: float) -> None:
        """Update link quality metrics."""
        try:
            link_key = (source_id, target_id)
            if link_key in self.links:
                link = self.links[link_key]
                link.latency = latency
                link.bandwidth = bandwidth
                link.reliability = reliability
                link.last_updated = time.time()
                
                # Update quality based on metrics
                link.quality = self._calculate_quality(latency, bandwidth, reliability)
                
                logger.debug(f"Updated link quality {source_id} -> {target_id}: {link.quality.value}")
            
        except Exception as e:
            logger.error(f"Error updating link quality: {e}")
    
    def _calculate_quality(self, latency: float, bandwidth: float, reliability: float) -> ConnectionQuality:
        """Calculate connection quality based on metrics."""
        try:
            # Quality scoring based on latency, bandwidth, and reliability
            if latency < 50 and bandwidth > 100 and reliability > 0.95:
                return ConnectionQuality.EXCELLENT
            elif latency < 100 and bandwidth > 50 and reliability > 0.9:
                return ConnectionQuality.GOOD
            elif latency < 200 and bandwidth > 10 and reliability > 0.8:
                return ConnectionQuality.FAIR
            elif latency < 500 and bandwidth > 1 and reliability > 0.7:
                return ConnectionQuality.POOR
            else:
                return ConnectionQuality.UNREACHABLE
                
        except Exception as e:
            logger.error(f"Error calculating quality: {e}")
            return ConnectionQuality.POOR
    
    def get_node_neighbors(self, node_id: str) -> List[str]:
        """Get neighbors of a node."""
        try:
            neighbors = []
            
            for (source_id, target_id), link in self.links.items():
                if source_id == node_id and link.is_active:
                    neighbors.append(target_id)
                elif target_id == node_id and link.is_active:
                    neighbors.append(source_id)
            
            return neighbors
            
        except Exception as e:
            logger.error(f"Error getting neighbors for node {node_id}: {e}")
            return []
    
    def get_topology_metrics(self) -> TopologyMetrics:
        """Get topology metrics."""
        try:
            total_nodes = len(self.nodes)
            active_nodes = sum(1 for node in self.nodes.values() if node.is_active)
            total_links = len(self.links)
            active_links = sum(1 for link in self.links.values() if link.is_active)
            
            # Calculate average metrics
            if active_links > 0:
                avg_latency = sum(link.latency for link in self.links.values() if link.is_active) / active_links
                avg_bandwidth = sum(link.bandwidth for link in self.links.values() if link.is_active) / active_links
            else:
                avg_latency = 0.0
                avg_bandwidth = 0.0
            
            # Calculate network diameter (simplified)
            diameter = self._calculate_diameter()
            
            # Calculate clustering coefficient
            clustering_coeff = self._calculate_clustering_coefficient()
            
            # Calculate connectivity ratio
            connectivity_ratio = self._calculate_connectivity_ratio()
            
            return TopologyMetrics(
                total_nodes=total_nodes,
                active_nodes=active_nodes,
                total_links=total_links,
                active_links=active_links,
                average_latency=avg_latency,
                average_bandwidth=avg_bandwidth,
                network_diameter=diameter,
                clustering_coefficient=clustering_coeff,
                connectivity_ratio=connectivity_ratio
            )
            
        except Exception as e:
            logger.error(f"Error calculating topology metrics: {e}")
            return TopologyMetrics()
    
    def _calculate_diameter(self) -> int:
        """Calculate network diameter (longest shortest path)."""
        try:
            if not self.nodes:
                return 0
            
            max_distance = 0
            
            # Use Floyd-Warshall algorithm for all-pairs shortest paths
            node_ids = list(self.nodes.keys())
            n = len(node_ids)
            
            # Initialize distance matrix
            dist = [[float('inf')] * n for _ in range(n)]
            
            # Set direct connections
            for i, node_id in enumerate(node_ids):
                dist[i][i] = 0
                neighbors = self.get_node_neighbors(node_id)
                for neighbor in neighbors:
                    if neighbor in node_ids:
                        j = node_ids.index(neighbor)
                        dist[i][j] = 1
            
            # Floyd-Warshall algorithm
            for k in range(n):
                for i in range(n):
                    for j in range(n):
                        if dist[i][k] + dist[k][j] < dist[i][j]:
                            dist[i][j] = dist[i][k] + dist[k][j]
            
            # Find maximum distance
            for i in range(n):
                for j in range(n):
                    if dist[i][j] != float('inf') and dist[i][j] > max_distance:
                        max_distance = int(dist[i][j])
            
            return max_distance
            
        except Exception as e:
            logger.error(f"Error calculating diameter: {e}")
            return 0
    
    def _calculate_clustering_coefficient(self) -> float:
        """Calculate average clustering coefficient."""
        try:
            if not self.nodes:
                return 0.0
            
            total_coefficient = 0.0
            valid_nodes = 0
            
            for node_id in self.nodes.keys():
                neighbors = self.get_node_neighbors(node_id)
                k = len(neighbors)
                
                if k < 2:
                    continue
                
                # Count edges between neighbors
                edges_between_neighbors = 0
                for i, neighbor1 in enumerate(neighbors):
                    for neighbor2 in neighbors[i+1:]:
                        if (neighbor1, neighbor2) in self.links or (neighbor2, neighbor1) in self.links:
                            edges_between_neighbors += 1
                
                # Calculate clustering coefficient for this node
                max_possible_edges = k * (k - 1) / 2
                if max_possible_edges > 0:
                    coefficient = edges_between_neighbors / max_possible_edges
                    total_coefficient += coefficient
                    valid_nodes += 1
            
            return total_coefficient / valid_nodes if valid_nodes > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating clustering coefficient: {e}")
            return 0.0
    
    def _calculate_connectivity_ratio(self) -> float:
        """Calculate connectivity ratio."""
        try:
            if not self.nodes:
                return 0.0
            
            n = len(self.nodes)
            max_possible_links = n * (n - 1) / 2
            actual_links = len(self.links)
            
            return actual_links / max_possible_links if max_possible_links > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating connectivity ratio: {e}")
            return 0.0

class TopologyOptimizer:
    """Optimizes network topology for better performance."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize topology optimizer."""
        self.config = config
        self.optimization_interval = config.get("optimization_interval", 300)  # 5 minutes
        self.target_topology_type = TopologyType(config.get("target_topology", "adaptive"))
        logger.info("Initialized topology optimizer")
    
    def optimize_topology(self, discovery: TopologyDiscovery) -> List[Tuple[str, str]]:
        """Optimize network topology."""
        try:
            recommendations = []
            
            if self.target_topology_type == TopologyType.MESH:
                recommendations.extend(self._optimize_for_mesh(discovery))
            elif self.target_topology_type == TopologyType.STAR:
                recommendations.extend(self._optimize_for_star(discovery))
            elif self.target_topology_type == TopologyType.RING:
                recommendations.extend(self._optimize_for_ring(discovery))
            elif self.target_topology_type == TopologyType.TREE:
                recommendations.extend(self._optimize_for_tree(discovery))
            elif self.target_topology_type == TopologyType.HYBRID:
                recommendations.extend(self._optimize_for_hybrid(discovery))
            elif self.target_topology_type == TopologyType.ADAPTIVE:
                recommendations.extend(self._optimize_adaptive(discovery))
            
            logger.info(f"Generated {len(recommendations)} topology optimization recommendations")
            return recommendations
            
        except Exception as e:
            logger.error(f"Error optimizing topology: {e}")
            return []
    
    def _optimize_for_mesh(self, discovery: TopologyDiscovery) -> List[Tuple[str, str]]:
        """Optimize for mesh topology."""
        try:
            recommendations = []
            nodes = list(discovery.nodes.keys())
            
            # Add missing connections for better mesh connectivity
            for i, node1 in enumerate(nodes):
                for node2 in nodes[i+1:]:
                    link_key1 = (node1, node2)
                    link_key2 = (node2, node1)
                    
                    if link_key1 not in discovery.links and link_key2 not in discovery.links:
                        recommendations.append(("add_link", node1, node2))
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error optimizing for mesh: {e}")
            return []
    
    def _optimize_for_star(self, discovery: TopologyDiscovery) -> List[Tuple[str, str]]:
        """Optimize for star topology."""
        try:
            recommendations = []
            nodes = list(discovery.nodes.keys())
            
            if not nodes:
                return recommendations
            
            # Find the best hub node (highest capability score)
            hub_node = self._find_best_hub_node(discovery)
            
            # Remove connections between non-hub nodes
            for node1 in nodes:
                if node1 == hub_node:
                    continue
                for node2 in nodes:
                    if node2 == hub_node or node2 == node1:
                        continue
                    
                    link_key1 = (node1, node2)
                    link_key2 = (node2, node1)
                    
                    if link_key1 in discovery.links or link_key2 in discovery.links:
                        recommendations.append(("remove_link", node1, node2))
            
            # Ensure all nodes connect to hub
            for node in nodes:
                if node != hub_node:
                    link_key1 = (hub_node, node)
                    link_key2 = (node, hub_node)
                    
                    if link_key1 not in discovery.links and link_key2 not in discovery.links:
                        recommendations.append(("add_link", hub_node, node))
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error optimizing for star: {e}")
            return []
    
    def _optimize_for_ring(self, discovery: TopologyDiscovery) -> List[Tuple[str, str]]:
        """Optimize for ring topology."""
        try:
            recommendations = []
            nodes = list(discovery.nodes.keys())
            
            if len(nodes) < 3:
                return recommendations
            
            # Create a ring by connecting each node to the next one
            for i, node in enumerate(nodes):
                next_node = nodes[(i + 1) % len(nodes)]
                link_key1 = (node, next_node)
                link_key2 = (next_node, node)
                
                if link_key1 not in discovery.links and link_key2 not in discovery.links:
                    recommendations.append(("add_link", node, next_node))
            
            # Remove extra connections
            for i, node1 in enumerate(nodes):
                for j, node2 in enumerate(nodes):
                    if j == i or j == (i + 1) % len(nodes) or j == (i - 1) % len(nodes):
                        continue
                    
                    link_key1 = (node1, node2)
                    link_key2 = (node2, node1)
                    
                    if link_key1 in discovery.links or link_key2 in discovery.links:
                        recommendations.append(("remove_link", node1, node2))
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error optimizing for ring: {e}")
            return []
    
    def _optimize_for_tree(self, discovery: TopologyDiscovery) -> List[Tuple[str, str]]:
        """Optimize for tree topology."""
        try:
            recommendations = []
            nodes = list(discovery.nodes.keys())
            
            if not nodes:
                return recommendations
            
            # Find root node (best hub)
            root_node = self._find_best_hub_node(discovery)
            
            # Build tree structure
            visited = {root_node}
            queue = deque([root_node])
            
            while queue:
                current = queue.popleft()
                neighbors = discovery.get_node_neighbors(current)
                
                for neighbor in neighbors:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
                        
                        # Ensure connection exists
                        link_key1 = (current, neighbor)
                        link_key2 = (neighbor, current)
                        
                        if link_key1 not in discovery.links and link_key2 not in discovery.links:
                            recommendations.append(("add_link", current, neighbor))
            
            # Remove cycles (extra connections)
            for node1 in nodes:
                for node2 in nodes:
                    if node1 == node2:
                        continue
                    
                    # Check if this connection creates a cycle
                    if self._would_create_cycle(discovery, node1, node2):
                        link_key1 = (node1, node2)
                        link_key2 = (node2, node1)
                        
                        if link_key1 in discovery.links or link_key2 in discovery.links:
                            recommendations.append(("remove_link", node1, node2))
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error optimizing for tree: {e}")
            return []
    
    def _optimize_for_hybrid(self, discovery: TopologyDiscovery) -> List[Tuple[str, str]]:
        """Optimize for hybrid topology."""
        try:
            recommendations = []
            
            # Combine mesh and star optimizations
            mesh_recs = self._optimize_for_mesh(discovery)
            star_recs = self._optimize_for_star(discovery)
            
            # Take some mesh connections and some star connections
            recommendations.extend(mesh_recs[:len(mesh_recs)//2])
            recommendations.extend(star_recs[:len(star_recs)//2])
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error optimizing for hybrid: {e}")
            return []
    
    def _optimize_adaptive(self, discovery: TopologyDiscovery) -> List[Tuple[str, str]]:
        """Adaptive optimization based on current topology."""
        try:
            metrics = discovery.get_topology_metrics()
            recommendations = []
            
            # Choose optimization strategy based on current metrics
            if metrics.connectivity_ratio < 0.3:
                # Low connectivity - optimize for mesh
                recommendations.extend(self._optimize_for_mesh(discovery))
            elif metrics.average_latency > 200:
                # High latency - optimize for star
                recommendations.extend(self._optimize_for_star(discovery))
            elif metrics.network_diameter > 5:
                # High diameter - optimize for ring
                recommendations.extend(self._optimize_for_ring(discovery))
            else:
                # Good metrics - optimize for tree
                recommendations.extend(self._optimize_for_tree(discovery))
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error in adaptive optimization: {e}")
            return []
    
    def _find_best_hub_node(self, discovery: TopologyDiscovery) -> str:
        """Find the best node to act as a hub."""
        try:
            if not discovery.nodes:
                return ""
            
            best_node = ""
            best_score = -1
            
            for node_id, node in discovery.nodes.items():
                # Calculate hub score based on capabilities and metrics
                score = 0
                
                # Capability score
                score += len(node.capabilities) * 10
                
                # Metrics score
                if "cpu_usage" in node.metrics:
                    score += (100 - node.metrics["cpu_usage"]) * 0.1
                if "memory_usage" in node.metrics:
                    score += (100 - node.metrics["memory_usage"]) * 0.1
                if "network_bandwidth" in node.metrics:
                    score += node.metrics["network_bandwidth"] * 0.01
                
                if score > best_score:
                    best_score = score
                    best_node = node_id
            
            return best_node
            
        except Exception as e:
            logger.error(f"Error finding best hub node: {e}")
            return list(discovery.nodes.keys())[0] if discovery.nodes else ""
    
    def _would_create_cycle(self, discovery: TopologyDiscovery, node1: str, node2: str) -> bool:
        """Check if adding a link would create a cycle."""
        try:
            # Simple cycle detection using DFS
            visited = set()
            stack = [node1]
            
            while stack:
                current = stack.pop()
                if current == node2:
                    return True
                
                if current in visited:
                    continue
                
                visited.add(current)
                neighbors = discovery.get_node_neighbors(current)
                
                for neighbor in neighbors:
                    if neighbor not in visited:
                        stack.append(neighbor)
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking for cycle: {e}")
            return False

class NetworkTopologyManager:
    """Main network topology management system."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize network topology manager."""
        self.config = config
        self.discovery = TopologyDiscovery(config.get("discovery", {}))
        self.optimizer = TopologyOptimizer(config.get("optimization", {}))
        
        # Topology maintenance
        self.maintenance_interval = config.get("maintenance_interval", 60)
        self.optimization_interval = config.get("optimization_interval", 300)
        
        # Background tasks
        self.maintenance_task: Optional[asyncio.Task] = None
        self.optimization_task: Optional[asyncio.Task] = None
        
        logger.info("Initialized network topology manager")
    
    async def start(self) -> None:
        """Start topology management."""
        try:
            # Start maintenance task
            self.maintenance_task = asyncio.create_task(self._maintenance_loop())
            
            # Start optimization task
            self.optimization_task = asyncio.create_task(self._optimization_loop())
            
            logger.info("Started network topology management")
            
        except Exception as e:
            logger.error(f"Error starting topology management: {e}")
    
    async def stop(self) -> None:
        """Stop topology management."""
        try:
            if self.maintenance_task:
                self.maintenance_task.cancel()
                try:
                    await self.maintenance_task
                except asyncio.CancelledError:
                    pass
            
            if self.optimization_task:
                self.optimization_task.cancel()
                try:
                    await self.optimization_task
                except asyncio.CancelledError:
                    pass
            
            logger.info("Stopped network topology management")
            
        except Exception as e:
            logger.error(f"Error stopping topology management: {e}")
    
    async def _maintenance_loop(self) -> None:
        """Maintenance loop for topology."""
        try:
            while True:
                await asyncio.sleep(self.maintenance_interval)
                await self._perform_maintenance()
                
        except asyncio.CancelledError:
            logger.info("Topology maintenance loop cancelled")
        except Exception as e:
            logger.error(f"Error in maintenance loop: {e}")
    
    async def _optimization_loop(self) -> None:
        """Optimization loop for topology."""
        try:
            while True:
                await asyncio.sleep(self.optimization_interval)
                await self._perform_optimization()
                
        except asyncio.CancelledError:
            logger.info("Topology optimization loop cancelled")
        except Exception as e:
            logger.error(f"Error in optimization loop: {e}")
    
    async def _perform_maintenance(self) -> None:
        """Perform topology maintenance."""
        try:
            current_time = time.time()
            
            # Remove inactive nodes
            inactive_nodes = [
                node_id for node_id, node in self.discovery.nodes.items()
                if current_time - node.last_seen > self.config.get("node_timeout", 300)
            ]
            
            for node_id in inactive_nodes:
                self.discovery.remove_node(node_id)
                logger.info(f"Removed inactive node {node_id}")
            
            # Remove inactive links
            inactive_links = [
                link_key for link_key, link in self.discovery.links.items()
                if current_time - link.last_updated > self.config.get("link_timeout", 60)
            ]
            
            for source_id, target_id in inactive_links:
                self.discovery.remove_link(source_id, target_id)
                logger.info(f"Removed inactive link {source_id} -> {target_id}")
            
            logger.debug("Completed topology maintenance")
            
        except Exception as e:
            logger.error(f"Error performing maintenance: {e}")
    
    async def _perform_optimization(self) -> None:
        """Perform topology optimization."""
        try:
            recommendations = self.optimizer.optimize_topology(self.discovery)
            
            for recommendation in recommendations:
                action, node1, node2 = recommendation
                
                if action == "add_link":
                    # Add new link
                    link = NetworkLink(
                        source_id=node1,
                        target_id=node2,
                        quality=ConnectionQuality.GOOD,
                        latency=0.0,
                        bandwidth=0.0,
                        reliability=1.0
                    )
                    self.discovery.add_link(link)
                    logger.info(f"Added optimization link {node1} -> {node2}")
                
                elif action == "remove_link":
                    # Remove existing link
                    self.discovery.remove_link(node1, node2)
                    logger.info(f"Removed optimization link {node1} -> {node2}")
            
            logger.debug("Completed topology optimization")
            
        except Exception as e:
            logger.error(f"Error performing optimization: {e}")
    
    def add_node(self, node: NetworkNode) -> None:
        """Add a node to the topology."""
        self.discovery.add_node(node)
    
    def remove_node(self, node_id: str) -> None:
        """Remove a node from the topology."""
        self.discovery.remove_node(node_id)
    
    def add_link(self, link: NetworkLink) -> None:
        """Add a link to the topology."""
        self.discovery.add_link(link)
    
    def remove_link(self, source_id: str, target_id: str) -> None:
        """Remove a link from the topology."""
        self.discovery.remove_link(source_id, target_id)
    
    def update_link_quality(self, source_id: str, target_id: str, 
                          latency: float, bandwidth: float, reliability: float) -> None:
        """Update link quality."""
        self.discovery.update_link_quality(source_id, target_id, latency, bandwidth, reliability)
    
    def get_topology_metrics(self) -> TopologyMetrics:
        """Get topology metrics."""
        return self.discovery.get_topology_metrics()
    
    def get_node_neighbors(self, node_id: str) -> List[str]:
        """Get neighbors of a node."""
        return self.discovery.get_node_neighbors(node_id)
    
    def get_shortest_path(self, source_id: str, target_id: str) -> List[str]:
        """Get shortest path between two nodes."""
        try:
            if source_id not in self.discovery.nodes or target_id not in self.discovery.nodes:
                return []
            
            # Use BFS to find shortest path
            queue = deque([(source_id, [source_id])])
            visited = {source_id}
            
            while queue:
                current, path = queue.popleft()
                
                if current == target_id:
                    return path
                
                neighbors = self.discovery.get_node_neighbors(current)
                for neighbor in neighbors:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, path + [neighbor]))
            
            return []  # No path found
            
        except Exception as e:
            logger.error(f"Error finding shortest path: {e}")
            return []

__all__ = [
    "NetworkTopologyManager",
    "TopologyDiscovery",
    "TopologyOptimizer",
    "NetworkTopology",
    "NetworkNode",
    "NetworkLink",
    "TopologyMetrics",
    "TopologyType",
    "NodeRole",
    "ConnectionQuality",
]