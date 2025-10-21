"""
Advanced Database Indexing System for DubChain

This module provides comprehensive indexing capabilities including:
- Multi-dimensional indexes
- Full-text search indexes
- Composite indexes
- Partial indexes
- Index optimization and maintenance
"""

import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
import threading
from collections import defaultdict

from ..errors import ClientError
from ..logging import get_logger

logger = get_logger(__name__)

class IndexType(Enum):
    """Types of indexes."""
    B_TREE = "btree"
    HASH = "hash"
    FULL_TEXT = "fulltext"
    COMPOSITE = "composite"
    PARTIAL = "partial"
    SPATIAL = "spatial"
    BITMAP = "bitmap"

class IndexStatus(Enum):
    """Index status."""
    BUILDING = "building"
    READY = "ready"
    MAINTENANCE = "maintenance"
    DROPPED = "dropped"
    ERROR = "error"

@dataclass
class IndexDefinition:
    """Index definition."""
    name: str
    table_name: str
    columns: List[str]
    index_type: IndexType
    unique: bool = False
    partial_condition: Optional[str] = None
    include_columns: List[str] = field(default_factory=list)
    options: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

@dataclass
class IndexStats:
    """Index statistics."""
    name: str
    table_name: str
    size_bytes: int
    row_count: int
    distinct_values: int
    selectivity: float
    last_used: Optional[float] = None
    usage_count: int = 0
    maintenance_count: int = 0

@dataclass
class IndexConfig:
    """Index configuration."""
    auto_maintenance: bool = True
    maintenance_interval: int = 3600  # seconds
    rebuild_threshold: float = 0.1  # 10% fragmentation
    stats_update_interval: int = 1800  # seconds
    max_indexes_per_table: int = 10
    enable_partial_indexes: bool = True
    enable_composite_indexes: bool = True

class IndexManager:
    """Main index manager."""
    
    def __init__(self, config: IndexConfig):
        """Initialize index manager."""
        self.config = config
        self.indexes: Dict[str, IndexDefinition] = {}
        self.index_stats: Dict[str, IndexStats] = {}
        self.maintenance_thread = None
        self.stats_thread = None
        self.running = False
        
        logger.info("Initialized index manager")
    
    def start(self) -> None:
        """Start index manager."""
        self.running = True
        
        if self.config.auto_maintenance:
            self.maintenance_thread = threading.Thread(target=self._maintenance_loop, daemon=True)
            self.maintenance_thread.start()
        
        self.stats_thread = threading.Thread(target=self._stats_update_loop, daemon=True)
        self.stats_thread.start()
        
        logger.info("Index manager started")
    
    def stop(self) -> None:
        """Stop index manager."""
        self.running = False
        
        if self.maintenance_thread:
            self.maintenance_thread.join(timeout=5)
        
        if self.stats_thread:
            self.stats_thread.join(timeout=5)
        
        logger.info("Index manager stopped")
    
    def create_index(self, definition: IndexDefinition) -> bool:
        """Create a new index."""
        try:
            # Validate index definition
            if not self._validate_index_definition(definition):
                return False
            
            # Check if index already exists
            if definition.name in self.indexes:
                logger.warning(f"Index {definition.name} already exists")
                return False
            
            # Create the index
            success = self._create_index_implementation(definition)
            
            if success:
                self.indexes[definition.name] = definition
                self.index_stats[definition.name] = IndexStats(
                    name=definition.name,
                    table_name=definition.table_name,
                    size_bytes=0,
                    row_count=0,
                    distinct_values=0,
                    selectivity=0.0
                )
                
                logger.info(f"Created index {definition.name} on {definition.table_name}")
                return True
            else:
                logger.error(f"Failed to create index {definition.name}")
                return False
                
        except Exception as e:
            logger.error(f"Error creating index {definition.name}: {e}")
            return False
    
    def drop_index(self, index_name: str) -> bool:
        """Drop an index."""
        try:
            if index_name not in self.indexes:
                logger.warning(f"Index {index_name} does not exist")
                return False
            
            definition = self.indexes[index_name]
            
            # Drop the index
            success = self._drop_index_implementation(definition)
            
            if success:
                del self.indexes[index_name]
                if index_name in self.index_stats:
                    del self.index_stats[index_name]
                
                logger.info(f"Dropped index {index_name}")
                return True
            else:
                logger.error(f"Failed to drop index {index_name}")
                return False
                
        except Exception as e:
            logger.error(f"Error dropping index {index_name}: {e}")
            return False
    
    def rebuild_index(self, index_name: str) -> bool:
        """Rebuild an index."""
        try:
            if index_name not in self.indexes:
                logger.warning(f"Index {index_name} does not exist")
                return False
            
            definition = self.indexes[index_name]
            
            logger.info(f"Rebuilding index {index_name}")
            
            # Rebuild the index
            success = self._rebuild_index_implementation(definition)
            
            if success:
                # Update stats
                if index_name in self.index_stats:
                    self.index_stats[index_name].maintenance_count += 1
                
                logger.info(f"Successfully rebuilt index {index_name}")
                return True
            else:
                logger.error(f"Failed to rebuild index {index_name}")
                return False
                
        except Exception as e:
            logger.error(f"Error rebuilding index {index_name}: {e}")
            return False
    
    def analyze_index(self, index_name: str) -> Optional[IndexStats]:
        """Analyze index statistics."""
        try:
            if index_name not in self.indexes:
                logger.warning(f"Index {index_name} does not exist")
                return None
            
            definition = self.indexes[index_name]
            
            # Analyze the index
            stats = self._analyze_index_implementation(definition)
            
            if stats:
                self.index_stats[index_name] = stats
                logger.info(f"Analyzed index {index_name}")
                return stats
            else:
                logger.error(f"Failed to analyze index {index_name}")
                return None
                
        except Exception as e:
            logger.error(f"Error analyzing index {index_name}: {e}")
            return None
    
    def get_index_recommendations(self, table_name: str, query_patterns: List[str]) -> List[IndexDefinition]:
        """Get index recommendations based on query patterns."""
        try:
            recommendations = []
            
            # Analyze query patterns
            column_usage = self._analyze_query_patterns(query_patterns)
            
            # Generate recommendations
            for columns, usage_count in column_usage.items():
                if usage_count > 5:  # Threshold for recommendation
                    # Check if index already exists
                    existing_index = self._find_existing_index(table_name, columns)
                    
                    if not existing_index:
                        recommendation = IndexDefinition(
                            name=f"idx_{table_name}_{'_'.join(columns)}",
                            table_name=table_name,
                            columns=columns,
                            index_type=IndexType.B_TREE,
                            unique=False
                        )
                        recommendations.append(recommendation)
            
            logger.info(f"Generated {len(recommendations)} index recommendations for {table_name}")
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating index recommendations: {e}")
            return []
    
    def optimize_indexes(self) -> Dict[str, Any]:
        """Optimize all indexes."""
        try:
            optimization_results = {
                "rebuilt": [],
                "analyzed": [],
                "dropped": [],
                "errors": []
            }
            
            for index_name in list(self.indexes.keys()):
                try:
                    # Analyze index
                    stats = self.analyze_index(index_name)
                    
                    if stats:
                        optimization_results["analyzed"].append(index_name)
                        
                        # Check if rebuild is needed
                        if stats.selectivity < 0.1:  # Low selectivity
                            if self.rebuild_index(index_name):
                                optimization_results["rebuilt"].append(index_name)
                            else:
                                optimization_results["errors"].append(f"Failed to rebuild {index_name}")
                        
                        # Check if index should be dropped
                        if stats.usage_count == 0 and stats.last_used and (time.time() - stats.last_used) > 86400:  # 24 hours
                            if self.drop_index(index_name):
                                optimization_results["dropped"].append(index_name)
                            else:
                                optimization_results["errors"].append(f"Failed to drop {index_name}")
                
                except Exception as e:
                    optimization_results["errors"].append(f"Error optimizing {index_name}: {e}")
            
            logger.info(f"Index optimization completed: {optimization_results}")
            return optimization_results
            
        except Exception as e:
            logger.error(f"Error during index optimization: {e}")
            return {"errors": [str(e)]}
    
    def _validate_index_definition(self, definition: IndexDefinition) -> bool:
        """Validate index definition."""
        try:
            # Check name
            if not definition.name or len(definition.name) > 64:
                logger.error("Invalid index name")
                return False
            
            # Check table name
            if not definition.table_name:
                logger.error("Table name is required")
                return False
            
            # Check columns
            if not definition.columns or len(definition.columns) == 0:
                logger.error("At least one column is required")
                return False
            
            # Check index type
            if definition.index_type not in IndexType:
                logger.error("Invalid index type")
                return False
            
            # Check partial index condition
            if definition.index_type == IndexType.PARTIAL and not definition.partial_condition:
                logger.error("Partial condition required for partial index")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating index definition: {e}")
            return False
    
    def _create_index_implementation(self, definition: IndexDefinition) -> bool:
        """Create index implementation."""
        try:
            # Generate SQL for index creation
            sql = self._generate_create_index_sql(definition)
            
            # Execute SQL (simplified - in real implementation would use actual database)
            logger.info(f"Executing SQL: {sql}")
            
            # Simulate index creation
            time.sleep(0.1)  # Simulate work
            
            return True
            
        except Exception as e:
            logger.error(f"Error creating index implementation: {e}")
            return False
    
    def _drop_index_implementation(self, definition: IndexDefinition) -> bool:
        """Drop index implementation."""
        try:
            # Generate SQL for index dropping
            sql = f"DROP INDEX IF EXISTS {definition.name}"
            
            # Execute SQL (simplified)
            logger.info(f"Executing SQL: {sql}")
            
            # Simulate index dropping
            time.sleep(0.05)  # Simulate work
            
            return True
            
        except Exception as e:
            logger.error(f"Error dropping index implementation: {e}")
            return False
    
    def _rebuild_index_implementation(self, definition: IndexDefinition) -> bool:
        """Rebuild index implementation."""
        try:
            # Generate SQL for index rebuild
            sql = f"REBUILD INDEX {definition.name}"
            
            # Execute SQL (simplified)
            logger.info(f"Executing SQL: {sql}")
            
            # Simulate index rebuild
            time.sleep(0.2)  # Simulate work
            
            return True
            
        except Exception as e:
            logger.error(f"Error rebuilding index implementation: {e}")
            return False
    
    def _analyze_index_implementation(self, definition: IndexDefinition) -> Optional[IndexStats]:
        """Analyze index implementation."""
        try:
            # Simulate index analysis
            import random
            
            stats = IndexStats(
                name=definition.name,
                table_name=definition.table_name,
                size_bytes=random.randint(1024, 1024*1024),  # 1KB to 1MB
                row_count=random.randint(100, 10000),
                distinct_values=random.randint(10, 1000),
                selectivity=random.uniform(0.1, 1.0),
                last_used=time.time() - random.randint(0, 3600),
                usage_count=random.randint(0, 100),
                maintenance_count=random.randint(0, 10)
            )
            
            return stats
            
        except Exception as e:
            logger.error(f"Error analyzing index implementation: {e}")
            return None
    
    def _generate_create_index_sql(self, definition: IndexDefinition) -> str:
        """Generate CREATE INDEX SQL."""
        try:
            sql_parts = ["CREATE"]
            
            if definition.unique:
                sql_parts.append("UNIQUE")
            
            sql_parts.append("INDEX")
            sql_parts.append(definition.name)
            sql_parts.append("ON")
            sql_parts.append(definition.table_name)
            
            # Add columns
            columns_str = "(" + ", ".join(definition.columns) + ")"
            sql_parts.append(columns_str)
            
            # Add partial condition
            if definition.partial_condition:
                sql_parts.append(f"WHERE {definition.partial_condition}")
            
            return " ".join(sql_parts)
            
        except Exception as e:
            logger.error(f"Error generating CREATE INDEX SQL: {e}")
            return ""
    
    def _analyze_query_patterns(self, query_patterns: List[str]) -> Dict[Tuple[str, ...], int]:
        """Analyze query patterns for column usage."""
        column_usage = defaultdict(int)
        
        for pattern in query_patterns:
            # Simple pattern analysis (in real implementation would be more sophisticated)
            if "WHERE" in pattern.upper():
                # Extract column names from WHERE clauses
                # This is simplified - real implementation would parse SQL properly
                words = pattern.split()
                for i, word in enumerate(words):
                    if word.upper() == "WHERE" and i + 1 < len(words):
                        column = words[i + 1].strip("(),")
                        if column.isalpha():
                            column_usage[(column,)] += 1
        
        return dict(column_usage)
    
    def _find_existing_index(self, table_name: str, columns: Tuple[str, ...]) -> Optional[IndexDefinition]:
        """Find existing index for table and columns."""
        for definition in self.indexes.values():
            if (definition.table_name == table_name and 
                tuple(definition.columns) == columns):
                return definition
        return None
    
    def _maintenance_loop(self) -> None:
        """Maintenance loop for automatic index maintenance."""
        while self.running:
            try:
                time.sleep(self.config.maintenance_interval)
                
                if self.running:
                    logger.info("Running automatic index maintenance")
                    self.optimize_indexes()
                    
            except Exception as e:
                logger.error(f"Error in maintenance loop: {e}")
    
    def _stats_update_loop(self) -> None:
        """Stats update loop for index statistics."""
        while self.running:
            try:
                time.sleep(self.config.stats_update_interval)
                
                if self.running:
                    logger.info("Updating index statistics")
                    for index_name in self.indexes.keys():
                        self.analyze_index(index_name)
                    
            except Exception as e:
                logger.error(f"Error in stats update loop: {e}")

class FullTextIndexManager:
    """Full-text search index manager."""
    
    def __init__(self, index_manager: IndexManager):
        """Initialize full-text index manager."""
        self.index_manager = index_manager
        self.fulltext_indexes: Dict[str, Dict[str, List[str]]] = {}
        logger.info("Initialized full-text index manager")
    
    def create_fulltext_index(self, table_name: str, column_name: str, content: List[str]) -> bool:
        """Create full-text search index."""
        try:
            if table_name not in self.fulltext_indexes:
                self.fulltext_indexes[table_name] = {}
            
            # Create inverted index
            inverted_index = self._build_inverted_index(content)
            self.fulltext_indexes[table_name][column_name] = inverted_index
            
            logger.info(f"Created full-text index for {table_name}.{column_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating full-text index: {e}")
            return False
    
    def search(self, table_name: str, column_name: str, query: str) -> List[int]:
        """Search full-text index."""
        try:
            if (table_name not in self.fulltext_indexes or 
                column_name not in self.fulltext_indexes[table_name]):
                return []
            
            inverted_index = self.fulltext_indexes[table_name][column_name]
            
            # Parse query
            terms = query.lower().split()
            
            # Find matching documents
            matching_docs = set()
            for term in terms:
                if term in inverted_index:
                    if not matching_docs:
                        matching_docs = set(inverted_index[term])
                    else:
                        matching_docs &= set(inverted_index[term])
            
            return list(matching_docs)
            
        except Exception as e:
            logger.error(f"Error searching full-text index: {e}")
            return []
    
    def _build_inverted_index(self, content: List[str]) -> Dict[str, List[int]]:
        """Build inverted index from content."""
        inverted_index = defaultdict(list)
        
        for doc_id, text in enumerate(content):
            # Simple tokenization
            words = text.lower().split()
            for word in words:
                # Remove punctuation
                word = ''.join(c for c in word if c.isalnum())
                if word:
                    inverted_index[word].append(doc_id)
        
        return dict(inverted_index)

class CompositeIndexManager:
    """Composite index manager."""
    
    def __init__(self, index_manager: IndexManager):
        """Initialize composite index manager."""
        self.index_manager = index_manager
        logger.info("Initialized composite index manager")
    
    def create_composite_index(self, table_name: str, columns: List[str], 
                              column_order: Optional[List[str]] = None) -> bool:
        """Create composite index."""
        try:
            if not column_order:
                column_order = columns
            
            definition = IndexDefinition(
                name=f"composite_{table_name}_{'_'.join(columns)}",
                table_name=table_name,
                columns=column_order,
                index_type=IndexType.COMPOSITE
            )
            
            return self.index_manager.create_index(definition)
            
        except Exception as e:
            logger.error(f"Error creating composite index: {e}")
            return False
    
    def optimize_composite_index(self, table_name: str, query_patterns: List[str]) -> List[IndexDefinition]:
        """Optimize composite index based on query patterns."""
        try:
            recommendations = []
            
            # Analyze column combinations
            column_combinations = self._analyze_column_combinations(query_patterns)
            
            for combination, usage_count in column_combinations.items():
                if usage_count > 3:  # Threshold for composite index
                    definition = IndexDefinition(
                        name=f"composite_{table_name}_{'_'.join(combination)}",
                        table_name=table_name,
                        columns=list(combination),
                        index_type=IndexType.COMPOSITE
                    )
                    recommendations.append(definition)
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error optimizing composite index: {e}")
            return []
    
    def _analyze_column_combinations(self, query_patterns: List[str]) -> Dict[Tuple[str, ...], int]:
        """Analyze column combinations in queries."""
        combinations = defaultdict(int)
        
        for pattern in query_patterns:
            # Extract multiple columns from WHERE clauses
            # Simplified implementation
            if "WHERE" in pattern.upper():
                words = pattern.split()
                columns = []
                in_where = False
                
                for word in words:
                    if word.upper() == "WHERE":
                        in_where = True
                        continue
                    elif in_where and word.upper() in ["AND", "OR"]:
                        continue
                    elif in_where and word.isalpha():
                        columns.append(word.strip("(),"))
                
                if len(columns) > 1:
                    combinations[tuple(sorted(columns))] += 1
        
        return dict(combinations)

__all__ = [
    "IndexManager",
    "FullTextIndexManager",
    "CompositeIndexManager",
    "IndexDefinition",
    "IndexStats",
    "IndexConfig",
    "IndexType",
    "IndexStatus",
]