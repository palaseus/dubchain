"""
Topology Optimization Module

This module provides network topology optimization using machine learning techniques.
"""

import numpy as np
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Set
from enum import Enum
import json
import random
from collections import defaultdict

from ..errors import ClientError
from dubchain.logging import get_logger

logger = get_logger(__name__)

class TopologyType(Enum):
    """Types of network topologies."""
    STAR = "star"
    RING = "ring"
    MESH = "mesh"
    TREE = "tree"
    HYBRID = "hybrid"
    CUSTOM = "custom"

class OptimizationMetric(Enum):
    """Metrics for topology optimization."""
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    RELIABILITY = "reliability"
    COST = "cost"
    SCALABILITY = "scalability"
    ENERGY_EFFICIENCY = "energy_efficiency"

@dataclass
class Node:
    """Network node representation."""
    id: str
    position: Tuple[float, float]
    capacity: float
    latency: float
    reliability: float
    cost: float
    energy_consumption: float
    node_type: str = "standard"

@dataclass
class Link:
    """Network link representation."""
    source: str
    target: str
    bandwidth: float
    latency: float
    reliability: float
    cost: float
    energy_consumption: float
    link_type: str = "standard"

@dataclass
class TopologyConfig:
    """Configuration for topology optimization."""
    optimization_metrics: List[OptimizationMetric]
    max_nodes: int = 100
    max_links_per_node: int = 10
    min_reliability: float = 0.95
    max_latency: float = 100.0
    budget_constraint: Optional[float] = None
    energy_constraint: Optional[float] = None

@dataclass
class TopologyResult:
    """Result of topology optimization."""
    nodes: List[Node]
    links: List[Link]
    topology_type: TopologyType
    optimization_score: float
    metrics: Dict[str, float]
    optimization_time: float
    iterations: int

class GraphAnalyzer:
    """Graph analysis utilities for topology optimization."""
    
    def __init__(self):
        """Initialize graph analyzer."""
        logger.info("Initialized GraphAnalyzer")
    
    def calculate_metrics(self, nodes: List[Node], links: List[Link]) -> Dict[str, float]:
        """Calculate topology metrics."""
        try:
            metrics = {}
            
            # Build adjacency list
            adjacency = defaultdict(list)
            for link in links:
                adjacency[link.source].append((link.target, link))
                adjacency[link.target].append((link.source, link))
            
            # Calculate various metrics
            metrics['avg_latency'] = self._calculate_avg_latency(nodes, links)
            metrics['throughput'] = self._calculate_throughput(nodes, links)
            metrics['reliability'] = self._calculate_reliability(nodes, links)
            metrics['cost'] = self._calculate_cost(nodes, links)
            metrics['energy_consumption'] = self._calculate_energy_consumption(nodes, links)
            metrics['connectivity'] = self._calculate_connectivity(nodes, adjacency)
            metrics['diameter'] = self._calculate_diameter(nodes, adjacency)
            metrics['clustering_coefficient'] = self._calculate_clustering_coefficient(nodes, adjacency)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating topology metrics: {e}")
            return {}
    
    def _calculate_avg_latency(self, nodes: List[Node], links: List[Link]) -> float:
        """Calculate average latency."""
        if not links:
            return 0.0
        
        total_latency = sum(link.latency for link in links)
        return total_latency / len(links)
    
    def _calculate_throughput(self, nodes: List[Node], links: List[Link]) -> float:
        """Calculate total throughput."""
        return sum(link.bandwidth for link in links)
    
    def _calculate_reliability(self, nodes: List[Node], links: List[Link]) -> float:
        """Calculate overall reliability."""
        if not links:
            return 1.0
        
        # Product of link reliabilities
        link_reliability = np.prod([link.reliability for link in links])
        
        # Product of node reliabilities
        node_reliability = np.prod([node.reliability for node in nodes])
        
        return link_reliability * node_reliability
    
    def _calculate_cost(self, nodes: List[Node], links: List[Link]) -> float:
        """Calculate total cost."""
        node_cost = sum(node.cost for node in nodes)
        link_cost = sum(link.cost for link in links)
        return node_cost + link_cost
    
    def _calculate_energy_consumption(self, nodes: List[Node], links: List[Link]) -> float:
        """Calculate total energy consumption."""
        node_energy = sum(node.energy_consumption for node in nodes)
        link_energy = sum(link.energy_consumption for link in links)
        return node_energy + link_energy
    
    def _calculate_connectivity(self, nodes: List[Node], adjacency: Dict[str, List]) -> float:
        """Calculate connectivity ratio."""
        if len(nodes) <= 1:
            return 1.0
        
        # Count connected components
        visited = set()
        components = 0
        
        for node in nodes:
            if node.id not in visited:
                self._dfs_connectivity(node.id, adjacency, visited)
                components += 1
        
        # Connectivity is inverse of number of components
        return 1.0 / components if components > 0 else 0.0
    
    def _dfs_connectivity(self, node_id: str, adjacency: Dict[str, List], visited: Set[str]) -> None:
        """DFS for connectivity calculation."""
        visited.add(node_id)
        for neighbor, _ in adjacency.get(node_id, []):
            if neighbor not in visited:
                self._dfs_connectivity(neighbor, adjacency, visited)
    
    def _calculate_diameter(self, nodes: List[Node], adjacency: Dict[str, List]) -> float:
        """Calculate network diameter."""
        if len(nodes) <= 1:
            return 0.0
        
        max_distance = 0
        
        for start_node in nodes:
            distances = self._bfs_distances(start_node.id, adjacency)
            max_distance = max(max_distance, max(distances.values()) if distances else 0)
        
        return max_distance
    
    def _bfs_distances(self, start_node: str, adjacency: Dict[str, List]) -> Dict[str, int]:
        """BFS to calculate distances from start node."""
        distances = {start_node: 0}
        queue = [start_node]
        
        while queue:
            current = queue.pop(0)
            for neighbor, _ in adjacency.get(current, []):
                if neighbor not in distances:
                    distances[neighbor] = distances[current] + 1
                    queue.append(neighbor)
        
        return distances
    
    def _calculate_clustering_coefficient(self, nodes: List[Node], adjacency: Dict[str, List]) -> float:
        """Calculate average clustering coefficient."""
        if not nodes:
            return 0.0
        
        total_coefficient = 0
        
        for node in nodes:
            neighbors = [neighbor for neighbor, _ in adjacency.get(node.id, [])]
            if len(neighbors) < 2:
                continue
            
            # Count edges between neighbors
            edges_between_neighbors = 0
            for i, neighbor1 in enumerate(neighbors):
                for neighbor2 in neighbors[i+1:]:
                    if any(n == neighbor2 for n, _ in adjacency.get(neighbor1, [])):
                        edges_between_neighbors += 1
            
            # Clustering coefficient for this node
            possible_edges = len(neighbors) * (len(neighbors) - 1) // 2
            if possible_edges > 0:
                coefficient = edges_between_neighbors / possible_edges
                total_coefficient += coefficient
        
        return total_coefficient / len(nodes)

class TopologyGenerator:
    """Generate different types of network topologies."""
    
    def __init__(self):
        """Initialize topology generator."""
        logger.info("Initialized TopologyGenerator")
    
    def generate_star_topology(self, node_count: int, center_position: Tuple[float, float] = (0.5, 0.5)) -> Tuple[List[Node], List[Link]]:
        """Generate star topology."""
        try:
            nodes = []
            links = []
            
            # Create center node
            center_node = Node(
                id="center",
                position=center_position,
                capacity=1000.0,
                latency=1.0,
                reliability=0.99,
                cost=100.0,
                energy_consumption=50.0,
                node_type="hub"
            )
            nodes.append(center_node)
            
            # Create peripheral nodes
            for i in range(node_count - 1):
                angle = 2 * np.pi * i / (node_count - 1)
                radius = 0.3
                x = center_position[0] + radius * np.cos(angle)
                y = center_position[1] + radius * np.sin(angle)
                
                node = Node(
                    id=f"node_{i}",
                    position=(x, y),
                    capacity=100.0,
                    latency=2.0,
                    reliability=0.95,
                    cost=50.0,
                    energy_consumption=25.0
                )
                nodes.append(node)
                
                # Create link to center
                link = Link(
                    source=node.id,
                    target=center_node.id,
                    bandwidth=100.0,
                    latency=5.0,
                    reliability=0.98,
                    cost=10.0,
                    energy_consumption=5.0
                )
                links.append(link)
            
            return nodes, links
            
        except Exception as e:
            logger.error(f"Error generating star topology: {e}")
            return [], []
    
    def generate_ring_topology(self, node_count: int) -> Tuple[List[Node], List[Link]]:
        """Generate ring topology."""
        try:
            nodes = []
            links = []
            
            # Create nodes in a circle
            for i in range(node_count):
                angle = 2 * np.pi * i / node_count
                radius = 0.4
                x = 0.5 + radius * np.cos(angle)
                y = 0.5 + radius * np.sin(angle)
                
                node = Node(
                    id=f"node_{i}",
                    position=(x, y),
                    capacity=100.0,
                    latency=2.0,
                    reliability=0.95,
                    cost=50.0,
                    energy_consumption=25.0
                )
                nodes.append(node)
            
            # Create ring links
            for i in range(node_count):
                next_i = (i + 1) % node_count
                link = Link(
                    source=f"node_{i}",
                    target=f"node_{next_i}",
                    bandwidth=100.0,
                    latency=3.0,
                    reliability=0.97,
                    cost=8.0,
                    energy_consumption=4.0
                )
                links.append(link)
            
            return nodes, links
            
        except Exception as e:
            logger.error(f"Error generating ring topology: {e}")
            return [], []
    
    def generate_mesh_topology(self, node_count: int, connection_probability: float = 0.3) -> Tuple[List[Node], List[Link]]:
        """Generate mesh topology."""
        try:
            nodes = []
            links = []
            
            # Create nodes randomly distributed
            for i in range(node_count):
                x = random.uniform(0.1, 0.9)
                y = random.uniform(0.1, 0.9)
                
                node = Node(
                    id=f"node_{i}",
                    position=(x, y),
                    capacity=100.0,
                    latency=2.0,
                    reliability=0.95,
                    cost=50.0,
                    energy_consumption=25.0
                )
                nodes.append(node)
            
            # Create links based on probability
            for i in range(node_count):
                for j in range(i + 1, node_count):
                    if random.random() < connection_probability:
                        # Calculate distance-based latency
                        pos1 = nodes[i].position
                        pos2 = nodes[j].position
                        distance = np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
                        latency = 1.0 + distance * 10.0
                        
                        link = Link(
                            source=nodes[i].id,
                            target=nodes[j].id,
                            bandwidth=100.0,
                            latency=latency,
                            reliability=0.96,
                            cost=5.0 + distance * 10.0,
                            energy_consumption=3.0 + distance * 5.0
                        )
                        links.append(link)
            
            return nodes, links
            
        except Exception as e:
            logger.error(f"Error generating mesh topology: {e}")
            return [], []
    
    def generate_tree_topology(self, levels: int, branching_factor: int) -> Tuple[List[Node], List[Link]]:
        """Generate tree topology."""
        try:
            nodes = []
            links = []
            node_id = 0
            
            # Create root node
            root_node = Node(
                id=f"node_{node_id}",
                position=(0.5, 0.9),
                capacity=200.0,
                latency=1.0,
                reliability=0.99,
                cost=100.0,
                energy_consumption=50.0,
                node_type="root"
            )
            nodes.append(root_node)
            node_id += 1
            
            current_level_nodes = [root_node]
            
            # Create levels
            for level in range(1, levels):
                next_level_nodes = []
                level_y = 0.9 - (level * 0.3)
                
                for parent_node in current_level_nodes:
                    for branch in range(branching_factor):
                        if node_id >= 50:  # Limit total nodes
                            break
                        
                        # Calculate position
                        parent_x = parent_node.position[0]
                        spacing = 0.2 / branching_factor
                        start_x = parent_x - 0.1 + spacing / 2
                        x = start_x + branch * spacing
                        
                        node = Node(
                            id=f"node_{node_id}",
                            position=(x, level_y),
                            capacity=100.0,
                            latency=2.0,
                            reliability=0.95,
                            cost=50.0,
                            energy_consumption=25.0
                        )
                        nodes.append(node)
                        next_level_nodes.append(node)
                        
                        # Create link to parent
                        link = Link(
                            source=node.id,
                            target=parent_node.id,
                            bandwidth=100.0,
                            latency=4.0,
                            reliability=0.97,
                            cost=8.0,
                            energy_consumption=4.0
                        )
                        links.append(link)
                        
                        node_id += 1
                
                current_level_nodes = next_level_nodes
            
            return nodes, links
            
        except Exception as e:
            logger.error(f"Error generating tree topology: {e}")
            return [], []

class TopologyOptimizer:
    """Main topology optimization engine."""
    
    def __init__(self, config: TopologyConfig):
        """Initialize topology optimizer."""
        self.config = config
        self.analyzer = GraphAnalyzer()
        self.generator = TopologyGenerator()
        
        logger.info("Initialized TopologyOptimizer")
    
    def optimize_topology(self, 
                         initial_nodes: List[Node], 
                         initial_links: List[Link]) -> TopologyResult:
        """Optimize network topology."""
        try:
            start_time = time.time()
            
            logger.info(f"Starting topology optimization with {len(initial_nodes)} nodes, {len(initial_links)} links")
            
            # Generate candidate topologies
            candidates = self._generate_candidate_topologies(initial_nodes)
            
            # Evaluate candidates
            best_topology = None
            best_score = float('-inf')
            iterations = 0
            
            for candidate_nodes, candidate_links in candidates:
                iterations += 1
                
                # Calculate metrics
                metrics = self.analyzer.calculate_metrics(candidate_nodes, candidate_links)
                
                # Calculate optimization score
                score = self._calculate_optimization_score(metrics)
                
                if score > best_score:
                    best_score = score
                    best_topology = (candidate_nodes, candidate_links, metrics)
                
                if iterations >= 50:  # Limit iterations
                    break
            
            if best_topology is None:
                raise ClientError("No valid topology found")
            
            best_nodes, best_links, best_metrics = best_topology
            optimization_time = time.time() - start_time
            
            # Determine topology type
            topology_type = self._classify_topology_type(best_nodes, best_links)
            
            result = TopologyResult(
                nodes=best_nodes,
                links=best_links,
                topology_type=topology_type,
                optimization_score=best_score,
                metrics=best_metrics,
                optimization_time=optimization_time,
                iterations=iterations
            )
            
            logger.info(f"Topology optimization completed: score = {best_score:.4f}, type = {topology_type.value}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in topology optimization: {e}")
            raise ClientError(f"Topology optimization failed: {str(e)}")
    
    def _generate_candidate_topologies(self, initial_nodes: List[Node]) -> List[Tuple[List[Node], List[Link]]]:
        """Generate candidate topologies for optimization."""
        candidates = []
        
        # Generate different topology types
        node_count = min(len(initial_nodes), self.config.max_nodes)
        
        # Star topology
        if node_count >= 2:
            star_nodes, star_links = self.generator.generate_star_topology(node_count)
            candidates.append((star_nodes, star_links))
        
        # Ring topology
        if node_count >= 3:
            ring_nodes, ring_links = self.generator.generate_ring_topology(node_count)
            candidates.append((ring_nodes, ring_links))
        
        # Mesh topology
        if node_count >= 4:
            mesh_nodes, mesh_links = self.generator.generate_mesh_topology(node_count, 0.3)
            candidates.append((mesh_nodes, mesh_links))
        
        # Tree topology
        if node_count >= 4:
            levels = min(3, int(np.log2(node_count)) + 1)
            branching_factor = min(3, node_count // levels)
            tree_nodes, tree_links = self.generator.generate_tree_topology(levels, branching_factor)
            candidates.append((tree_nodes, tree_links))
        
        # Random variations
        for _ in range(10):
            random_nodes, random_links = self._generate_random_topology(initial_nodes)
            candidates.append((random_nodes, random_links))
        
        return candidates
    
    def _generate_random_topology(self, initial_nodes: List[Node]) -> Tuple[List[Node], List[Link]]:
        """Generate random topology variation."""
        try:
            nodes = []
            links = []
            
            # Use initial nodes as base
            for node in initial_nodes[:self.config.max_nodes]:
                # Add some randomness to positions
                new_x = max(0.1, min(0.9, node.position[0] + random.uniform(-0.1, 0.1)))
                new_y = max(0.1, min(0.9, node.position[1] + random.uniform(-0.1, 0.1)))
                
                new_node = Node(
                    id=node.id,
                    position=(new_x, new_y),
                    capacity=node.capacity,
                    latency=node.latency,
                    reliability=node.reliability,
                    cost=node.cost,
                    energy_consumption=node.energy_consumption,
                    node_type=node.node_type
                )
                nodes.append(new_node)
            
            # Create random links
            max_links = min(len(nodes) * self.config.max_links_per_node, len(nodes) * (len(nodes) - 1) // 2)
            num_links = random.randint(len(nodes) - 1, max_links)
            
            created_links = set()
            for _ in range(num_links):
                source_idx = random.randint(0, len(nodes) - 1)
                target_idx = random.randint(0, len(nodes) - 1)
                
                if source_idx != target_idx:
                    link_key = tuple(sorted([nodes[source_idx].id, nodes[target_idx].id]))
                    if link_key not in created_links:
                        created_links.add(link_key)
                        
                        # Calculate distance-based properties
                        pos1 = nodes[source_idx].position
                        pos2 = nodes[target_idx].position
                        distance = np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
                        
                        link = Link(
                            source=nodes[source_idx].id,
                            target=nodes[target_idx].id,
                            bandwidth=random.uniform(50.0, 200.0),
                            latency=1.0 + distance * 10.0,
                            reliability=random.uniform(0.9, 0.99),
                            cost=5.0 + distance * 10.0,
                            energy_consumption=3.0 + distance * 5.0
                        )
                        links.append(link)
            
            return nodes, links
            
        except Exception as e:
            logger.error(f"Error generating random topology: {e}")
            return [], []
    
    def _calculate_optimization_score(self, metrics: Dict[str, float]) -> float:
        """Calculate optimization score based on configured metrics."""
        try:
            score = 0.0
            weight_sum = 0.0
            
            # Define weights for different metrics
            weights = {
                OptimizationMetric.LATENCY: -0.3,  # Lower is better
                OptimizationMetric.THROUGHPUT: 0.2,   # Higher is better
                OptimizationMetric.RELIABILITY: 0.3,  # Higher is better
                OptimizationMetric.COST: -0.1,        # Lower is better
                OptimizationMetric.SCALABILITY: 0.1,  # Higher is better
                OptimizationMetric.ENERGY_EFFICIENCY: -0.1  # Lower is better
            }
            
            # Map metrics to optimization metrics
            metric_mapping = {
                'avg_latency': OptimizationMetric.LATENCY,
                'throughput': OptimizationMetric.THROUGHPUT,
                'reliability': OptimizationMetric.RELIABILITY,
                'cost': OptimizationMetric.COST,
                'connectivity': OptimizationMetric.SCALABILITY,
                'energy_consumption': OptimizationMetric.ENERGY_EFFICIENCY
            }
            
            for metric_name, value in metrics.items():
                if metric_name in metric_mapping:
                    opt_metric = metric_mapping[metric_name]
                    if opt_metric in self.config.optimization_metrics:
                        weight = weights.get(opt_metric, 0.0)
                        score += weight * value
                        weight_sum += abs(weight)
            
            # Normalize by weight sum
            if weight_sum > 0:
                score = score / weight_sum
            
            return score
            
        except Exception as e:
            logger.error(f"Error calculating optimization score: {e}")
            return 0.0
    
    def _classify_topology_type(self, nodes: List[Node], links: List[Link]) -> TopologyType:
        """Classify the type of topology."""
        try:
            if not nodes or not links:
                return TopologyType.CUSTOM
            
            # Build adjacency list
            adjacency = defaultdict(list)
            for link in links:
                adjacency[link.source].append(link.target)
                adjacency[link.target].append(link.source)
            
            # Check for star topology (one node with many connections)
            max_connections = max(len(adjacency[node.id]) for node in nodes)
            if max_connections > len(nodes) * 0.7:
                return TopologyType.STAR
            
            # Check for ring topology (each node has exactly 2 connections)
            if all(len(adjacency[node.id]) == 2 for node in nodes):
                return TopologyType.RING
            
            # Check for tree topology (no cycles, connected)
            if self._is_tree(nodes, adjacency):
                return TopologyType.TREE
            
            # Check for mesh topology (high connectivity)
            avg_connections = sum(len(adjacency[node.id]) for node in nodes) / len(nodes)
            if avg_connections > len(nodes) * 0.3:
                return TopologyType.MESH
            
            return TopologyType.HYBRID
            
        except Exception as e:
            logger.error(f"Error classifying topology type: {e}")
            return TopologyType.CUSTOM
    
    def _is_tree(self, nodes: List[Node], adjacency: Dict[str, List]) -> bool:
        """Check if topology is a tree (connected, no cycles)."""
        try:
            if not nodes:
                return False
            
            # Check connectivity
            visited = set()
            self._dfs_tree_check(nodes[0].id, adjacency, visited, None)
            if len(visited) != len(nodes):
                return False
            
            # Check for cycles
            return not self._has_cycle(nodes, adjacency)
            
        except Exception as e:
            logger.error(f"Error checking if topology is tree: {e}")
            return False
    
    def _dfs_tree_check(self, node_id: str, adjacency: Dict[str, List], visited: Set[str], parent: Optional[str]) -> None:
        """DFS for tree connectivity check."""
        visited.add(node_id)
        for neighbor in adjacency.get(node_id, []):
            if neighbor != parent:
                if neighbor in visited:
                    return  # Cycle detected
                self._dfs_tree_check(neighbor, adjacency, visited, node_id)
    
    def _has_cycle(self, nodes: List[Node], adjacency: Dict[str, List]) -> bool:
        """Check if topology has cycles."""
        try:
            visited = set()
            rec_stack = set()
            
            for node in nodes:
                if node.id not in visited:
                    if self._dfs_cycle_check(node.id, adjacency, visited, rec_stack):
                        return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking for cycles: {e}")
            return False
    
    def _dfs_cycle_check(self, node_id: str, adjacency: Dict[str, List], visited: Set[str], rec_stack: Set[str]) -> bool:
        """DFS for cycle detection."""
        visited.add(node_id)
        rec_stack.add(node_id)
        
        for neighbor in adjacency.get(node_id, []):
            if neighbor not in visited:
                if self._dfs_cycle_check(neighbor, adjacency, visited, rec_stack):
                    return True
            elif neighbor in rec_stack:
                return True
        
        rec_stack.remove(node_id)
        return False

__all__ = [
    "TopologyOptimizer",
    "GraphAnalyzer",
    "TopologyGenerator",
    "TopologyType",
    "OptimizationMetric",
    "Node",
    "Link",
    "TopologyConfig",
    "TopologyResult",
]