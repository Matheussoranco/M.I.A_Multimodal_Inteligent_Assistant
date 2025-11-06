import asyncio
import hashlib
import json
import logging
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple, Union

try:
    import numpy as np
except ImportError:
    np = None  # Fallback for when numpy is not available

from ..providers import provider_registry

logger = logging.getLogger(__name__)


@dataclass
class KnowledgeNode:
    """Represents a node in the knowledge graph."""

    id: str
    content: Any
    node_type: str
    embedding: Optional[Any] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    confidence: float = 1.0
    access_count: int = 0
    last_accessed: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "content": self.content,
            "node_type": self.node_type,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "confidence": self.confidence,
            "access_count": self.access_count,
            "last_accessed": (
                self.last_accessed.isoformat() if self.last_accessed else None
            ),
        }


@dataclass
class KnowledgeEdge:
    """Represents an edge/relationship in the knowledge graph."""

    source_id: str
    target_id: str
    relationship_type: str
    weight: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    last_accessed: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "relationship_type": self.relationship_type,
            "weight": self.weight,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "access_count": self.access_count,
            "last_accessed": (
                self.last_accessed.isoformat() if self.last_accessed else None
            ),
        }


@dataclass
class VectorSearchResult:
    """Result of a vector similarity search."""

    node: KnowledgeNode
    similarity_score: float
    context_path: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "node": self.node.to_dict(),
            "similarity_score": self.similarity_score,
            "context_path": self.context_path,
        }


@dataclass
class GraphTraversalResult:
    """Result of a graph traversal operation."""

    path: List[KnowledgeNode]
    relationships: List[KnowledgeEdge]
    total_weight: float
    depth: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "path": [node.to_dict() for node in self.path],
            "relationships": [edge.to_dict() for edge in self.relationships],
            "total_weight": self.total_weight,
            "depth": self.depth,
        }


class KnowledgeMemoryGraph:
    """
    Hybrid knowledge memory graph.
    """

    def __init__(
        self,
        config_manager=None,
        *,
        embedding_provider=None,
        max_nodes: int = 10000,
        max_edges_per_node: int = 50,
        similarity_threshold: float = 0.7,
        decay_factor: float = 0.95,
        consolidation_threshold: float = 0.9,
        enable_learning: bool = True,
    ):
        self.config_manager = config_manager
        self.embedding_provider = (
            embedding_provider or self._get_default_embedding_provider()
        )
        self.max_nodes = max_nodes
        self.max_edges_per_node = max_edges_per_node
        self.similarity_threshold = similarity_threshold
        self.decay_factor = decay_factor
        self.consolidation_threshold = consolidation_threshold
        self.enable_learning = enable_learning

        # Core storage
        self.nodes: Dict[str, KnowledgeNode] = {}
        self.edges: Dict[Tuple[str, str], KnowledgeEdge] = {}
        self.node_embeddings: Dict[str, Any] = {}

        # Indexes for efficient access
        self.node_type_index: Dict[str, Set[str]] = defaultdict(set)
        self.relationship_index: Dict[str, Set[Tuple[str, str]]] = defaultdict(
            set
        )
        self.temporal_index: Dict[str, List[str]] = defaultdict(
            list
        )  # date -> node_ids

        # Learning and evolution
        self.knowledge_patterns: Dict[str, Dict[str, Any]] = {}
        self.consolidation_queue = deque(maxlen=1000)
        self.evolution_history: List[Dict[str, Any]] = []

        # Thread safety
        self.lock = threading.RLock()

        # Background processing
        self.consolidation_thread: Optional[threading.Thread] = None
        self.decay_thread: Optional[threading.Thread] = None
        self.processing_active = False

        logger.info(
            "Knowledge Memory Graph initialized with capacity for %d nodes",
            max_nodes,
        )

    def _get_default_embedding_provider(self):
        """Get default embedding provider."""
        try:
            # Try to get embedding provider from registry
            # For now, return None and use mock embeddings
            return None
        except Exception:
            logger.warning(
                "No embedding provider found, using mock embeddings"
            )
            return None

    def start_background_processing(self):
        """Start background knowledge processing."""
        if self.processing_active:
            return

        self.processing_active = True
        self.consolidation_thread = threading.Thread(
            target=self._consolidation_worker,
            daemon=True,
            name="KnowledgeConsolidation",
        )
        self.decay_thread = threading.Thread(
            target=self._decay_worker, daemon=True, name="KnowledgeDecay"
        )

        self.consolidation_thread.start()
        self.decay_thread.start()
        logger.info("Started background knowledge processing")

    def stop_background_processing(self):
        """Stop background knowledge processing."""
        self.processing_active = False
        if self.consolidation_thread:
            self.consolidation_thread.join(timeout=5.0)
        if self.decay_thread:
            self.decay_thread.join(timeout=5.0)
        logger.info("Stopped background knowledge processing")

    def add_knowledge(
        self,
        content: Any,
        node_type: str,
        metadata: Optional[Dict[str, Any]] = None,
        relationships: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """
        Add new knowledge to the graph.

        Args:
            content: The knowledge content
            node_type: Type of knowledge node
            metadata: Additional metadata
            relationships: List of relationships to establish

        Returns:
            Node ID of the added knowledge
        """
        with self.lock:
            # Generate node ID
            content_hash = hashlib.md5(str(content).encode()).hexdigest()[:16]
            node_id = f"{node_type}_{content_hash}"

            # Check for existing similar knowledge
            if node_id in self.nodes:
                # Update existing node
                existing = self.nodes[node_id]
                existing.content = content
                existing.metadata.update(metadata or {})
                existing.updated_at = datetime.now()
                existing.confidence = min(1.0, existing.confidence + 0.1)
                logger.debug("Updated existing knowledge node: %s", node_id)
            else:
                # Create new node
                node = KnowledgeNode(
                    id=node_id,
                    content=content,
                    node_type=node_type,
                    metadata=metadata or {},
                    confidence=0.8,  # Initial confidence
                )
                self.nodes[node_id] = node
                self.node_type_index[node_type].add(node_id)

                # Generate embedding
                self._generate_embedding(node)

                # Add to temporal index
                date_key = datetime.now().strftime("%Y-%m-%d")
                self.temporal_index[date_key].append(node_id)

                logger.debug("Added new knowledge node: %s", node_id)

            # Establish relationships
            if relationships:
                for rel in relationships:
                    self._add_relationship(node_id, rel)

            # Check for consolidation opportunities
            if self.enable_learning:
                self.consolidation_queue.append(node_id)

            # Enforce capacity limits
            self._enforce_capacity_limits()

            return node_id

    def _generate_embedding(self, node: KnowledgeNode):
        """Generate embedding for a knowledge node."""
        try:
            if self.embedding_provider:
                # Use real embedding provider
                text_content = self._extract_text_content(node.content)
                embedding = self.embedding_provider.encode(text_content)
                if np is not None and isinstance(embedding, np.ndarray):
                    node.embedding = embedding
                    self.node_embeddings[node.id] = embedding
                elif embedding is not None:
                    node.embedding = embedding
                    self.node_embeddings[node.id] = embedding
            else:
                # Generate mock embedding for testing
                if np is not None:
                    node.embedding = np.random.rand(384).astype(np.float32)
                else:
                    node.embedding = [0.1] * 384  # Simple fallback
                self.node_embeddings[node.id] = node.embedding

        except Exception as exc:
            logger.debug(
                "Failed to generate embedding for node %s: %s", node.id, exc
            )
            # Fallback to random embedding
            if np is not None:
                node.embedding = np.random.rand(384).astype(np.float32)
            else:
                node.embedding = [0.1] * 384  # Simple fallback
            self.node_embeddings[node.id] = node.embedding

    def _extract_text_content(self, content: Any) -> str:
        """Extract text content from various content types."""
        if isinstance(content, str):
            return content
        elif isinstance(content, dict):
            # Try common text fields
            for field in ["text", "content", "description", "summary"]:
                if field in content and isinstance(content[field], str):
                    return content[field]
            return json.dumps(content)
        elif isinstance(content, list):
            return " ".join(str(item) for item in content)
        else:
            return str(content)

    def _add_relationship(self, source_id: str, relationship: Dict[str, Any]):
        """Add a relationship between nodes."""
        target_id = relationship.get("target_id")
        rel_type = relationship.get("type", "related")
        weight = relationship.get("weight", 1.0)
        metadata = relationship.get("metadata", {})

        if not target_id or target_id not in self.nodes:
            logger.debug(
                "Cannot add relationship: target %s not found", target_id
            )
            return

        edge_key = (source_id, target_id)
        if edge_key in self.edges:
            # Update existing edge
            edge = self.edges[edge_key]
            edge.weight = max(edge.weight, weight)
            edge.metadata.update(metadata)
        else:
            # Create new edge
            edge = KnowledgeEdge(
                source_id=source_id,
                target_id=target_id,
                relationship_type=rel_type,
                weight=weight,
                metadata=metadata,
            )
            self.edges[edge_key] = edge
            self.relationship_index[rel_type].add(edge_key)

        # Add reverse relationship if bidirectional
        if relationship.get("bidirectional", False):
            reverse_key = (target_id, source_id)
            if reverse_key not in self.edges:
                reverse_edge = KnowledgeEdge(
                    source_id=target_id,
                    target_id=source_id,
                    relationship_type=f"reverse_{rel_type}",
                    weight=weight,
                    metadata=metadata,
                )
                self.edges[reverse_key] = reverse_edge
                self.relationship_index[f"reverse_{rel_type}"].add(reverse_key)

    def search_knowledge(
        self,
        query: str,
        node_types: Optional[List[str]] = None,
        limit: int = 10,
        include_context: bool = True,
    ) -> List[VectorSearchResult]:
        """
        Search knowledge using vector similarity.

        Args:
            query: Search query
            node_types: Filter by node types
            limit: Maximum results to return
            include_context: Include contextual relationships

        Returns:
            List of search results with similarity scores
        """
        with self.lock:
            # Generate query embedding
            query_embedding = self._generate_query_embedding(query)
            if query_embedding is None:
                return []

            # Filter candidate nodes
            candidate_ids = set()
            if node_types:
                for node_type in node_types:
                    candidate_ids.update(
                        self.node_type_index.get(node_type, set())
                    )
            else:
                candidate_ids = set(self.nodes.keys())

            # Calculate similarities
            results = []
            for node_id in candidate_ids:
                if node_id in self.node_embeddings:
                    node_embedding = self.node_embeddings[node_id]
                    similarity = self._cosine_similarity(
                        query_embedding, node_embedding
                    )

                    if similarity >= self.similarity_threshold:
                        node = self.nodes[node_id]
                        context_path = []

                        if include_context:
                            context_path = self._get_context_path(
                                node_id, query_embedding
                            )

                        result = VectorSearchResult(
                            node=node,
                            similarity_score=similarity,
                            context_path=context_path,
                        )
                        results.append(result)

                        # Update access statistics
                        node.access_count += 1
                        node.last_accessed = datetime.now()

            # Sort by similarity and return top results
            results.sort(key=lambda x: x.similarity_score, reverse=True)
            return results[:limit]

    def _generate_query_embedding(self, query: str) -> Optional[Any]:
        """Generate embedding for search query."""
        try:
            if self.embedding_provider:
                return self.embedding_provider.encode(query)
            else:
                # Mock embedding
                if np is not None:
                    return np.random.rand(384).astype(np.float32)
                else:
                    return [0.1] * 384  # Simple fallback
        except Exception as exc:
            logger.debug("Failed to generate query embedding: %s", exc)
            return None

    def _cosine_similarity(self, a: Any, b: Any) -> float:
        """Calculate cosine similarity between two vectors."""
        try:
            if np is not None:
                dot_product = np.dot(a, b)
                norm_a = np.linalg.norm(a)
                norm_b = np.linalg.norm(b)
                return (
                    dot_product / (norm_a * norm_b)
                    if norm_a > 0 and norm_b > 0
                    else 0.0
                )
            else:
                # Simple dot product fallback
                if (
                    isinstance(a, list)
                    and isinstance(b, list)
                    and len(a) == len(b)
                ):
                    dot_product = sum(x * y for x, y in zip(a, b))
                    norm_a = sum(x * x for x in a) ** 0.5
                    norm_b = sum(x * x for x in b) ** 0.5
                    return (
                        dot_product / (norm_a * norm_b)
                        if norm_a > 0 and norm_b > 0
                        else 0.0
                    )
                return 0.0
        except Exception:
            return 0.0

    def _get_context_path(
        self, node_id: str, query_embedding: Any
    ) -> List[str]:
        """Get contextual path for a node based on relationships."""
        context_nodes = []
        visited = set([node_id])

        # Find related nodes with high similarity
        for edge_key, edge in self.edges.items():
            if node_id in edge_key:
                other_id = (
                    edge.target_id
                    if edge.source_id == node_id
                    else edge.source_id
                )
                if (
                    other_id not in visited
                    and other_id in self.node_embeddings
                ):
                    other_embedding = self.node_embeddings[other_id]
                    similarity = self._cosine_similarity(
                        query_embedding, other_embedding
                    )
                    if (
                        similarity >= self.similarity_threshold * 0.8
                    ):  # Slightly lower threshold for context
                        context_nodes.append(other_id)
                        visited.add(other_id)

        return context_nodes[:3]  # Limit context nodes

    def traverse_relationships(
        self,
        start_node_id: str,
        relationship_types: Optional[List[str]] = None,
        max_depth: int = 3,
        max_nodes: int = 20,
    ) -> List[GraphTraversalResult]:
        """
        Traverse graph relationships from a starting node.

        Args:
            start_node_id: Starting node ID
            relationship_types: Types of relationships to traverse
            max_depth: Maximum traversal depth
            max_nodes: Maximum nodes to visit

        Returns:
            List of traversal results
        """
        with self.lock:
            if start_node_id not in self.nodes:
                return []

            results = []
            visited = set()
            queue = deque(
                [(start_node_id, [], [], 0, 0.0)]
            )  # node_id, path, relationships, depth, total_weight

            while queue and len(results) < max_nodes:
                current_id, path, relationships, depth, total_weight = (
                    queue.popleft()
                )

                if current_id in visited or depth > max_depth:
                    continue

                visited.add(current_id)
                current_node = self.nodes[current_id]
                current_path = path + [current_node]

                # Check if this is a valid result (depth > 0)
                if depth > 0:
                    result = GraphTraversalResult(
                        path=current_path,
                        relationships=relationships,
                        total_weight=total_weight,
                        depth=depth,
                    )
                    results.append(result)

                # Explore neighbors
                if depth < max_depth:
                    neighbors = self._get_node_neighbors(
                        current_id, relationship_types
                    )
                    for neighbor_id, edge in neighbors:
                        if neighbor_id not in visited:
                            new_relationships = relationships + [edge]
                            new_weight = total_weight + edge.weight
                            queue.append(
                                (
                                    neighbor_id,
                                    current_path,
                                    new_relationships,
                                    depth + 1,
                                    new_weight,
                                )
                            )

            return results

    def _get_node_neighbors(
        self, node_id: str, relationship_types: Optional[List[str]] = None
    ) -> List[Tuple[str, KnowledgeEdge]]:
        """Get neighboring nodes for a given node."""
        neighbors = []

        for edge_key, edge in self.edges.items():
            if node_id in edge_key:
                other_id = (
                    edge.target_id
                    if edge.source_id == node_id
                    else edge.source_id
                )

                # Filter by relationship type if specified
                if (
                    relationship_types
                    and edge.relationship_type not in relationship_types
                ):
                    continue

                neighbors.append((other_id, edge))

        # Sort by edge weight
        neighbors.sort(key=lambda x: x[1].weight, reverse=True)
        return neighbors[: self.max_edges_per_node]

    def consolidate_knowledge(self):
        """Consolidate similar knowledge nodes."""
        with self.lock:
            consolidation_candidates = self._find_consolidation_candidates()

            for group in consolidation_candidates:
                if len(group) > 1:
                    self._merge_knowledge_nodes(group)

    def _find_consolidation_candidates(self) -> List[List[str]]:
        """Find groups of similar knowledge nodes for consolidation."""
        candidates = []

        # Group nodes by type
        type_groups = defaultdict(list)
        for node_id, node in self.nodes.items():
            if node.embedding is not None:
                type_groups[node.node_type].append((node_id, node.embedding))

        # Find similar nodes within each type
        for node_type, nodes_with_embeddings in type_groups.items():
            if len(nodes_with_embeddings) < 2:
                continue

            if np is not None:
                embeddings = np.array(
                    [emb for _, emb in nodes_with_embeddings]
                )
                similarities = np.dot(embeddings, embeddings.T)
            else:
                # Simple similarity calculation fallback
                embeddings = [emb for _, emb in nodes_with_embeddings]
                n = len(embeddings)
                similarities = [[0.0] * n for _ in range(n)]
                for i in range(n):
                    for j in range(n):
                        similarities[i][j] = self._cosine_similarity(
                            embeddings[i][1], embeddings[j][1]
                        )

            # Find clusters of similar nodes
            visited = set()
            for i, (node_id, _) in enumerate(nodes_with_embeddings):
                if node_id in visited:
                    continue

                cluster = [node_id]
                visited.add(node_id)

                for j, (other_id, _) in enumerate(nodes_with_embeddings):
                    if (
                        other_id not in visited
                        and similarities[i][j] >= self.consolidation_threshold
                    ):
                        cluster.append(other_id)
                        visited.add(other_id)

                if len(cluster) > 1:
                    candidates.append(cluster)

        return candidates

    def _merge_knowledge_nodes(self, node_ids: List[str]):
        """Merge multiple knowledge nodes into one."""
        if not node_ids:
            return

        # Choose the node with highest confidence as primary
        primary_id = max(node_ids, key=lambda nid: self.nodes[nid].confidence)
        primary_node = self.nodes[primary_id]

        # Merge content and metadata
        merged_content = primary_node.content
        merged_metadata = primary_node.metadata.copy()

        # Update relationships
        for node_id in node_ids:
            if node_id == primary_id:
                continue

            node = self.nodes[node_id]

            # Merge metadata
            for key, value in node.metadata.items():
                if key not in merged_metadata:
                    merged_metadata[key] = value
                elif isinstance(merged_metadata[key], list) and isinstance(
                    value, list
                ):
                    merged_metadata[key].extend(value)
                elif isinstance(merged_metadata[key], dict) and isinstance(
                    value, dict
                ):
                    merged_metadata[key].update(value)

            # Transfer relationships
            for edge_key, edge in list(self.edges.items()):
                if node_id in edge_key:
                    new_edge_key = (
                        (
                            primary_id
                            if edge.source_id == node_id
                            else edge.source_id
                        ),
                        (
                            primary_id
                            if edge.target_id == node_id
                            else edge.target_id
                        ),
                    )

                    if new_edge_key[0] != new_edge_key[1]:  # Avoid self-loops
                        if new_edge_key in self.edges:
                            # Merge edge weights
                            existing_edge = self.edges[new_edge_key]
                            existing_edge.weight = max(
                                existing_edge.weight, edge.weight
                            )
                        else:
                            # Create new edge
                            new_edge = KnowledgeEdge(
                                source_id=new_edge_key[0],
                                target_id=new_edge_key[1],
                                relationship_type=edge.relationship_type,
                                weight=edge.weight,
                                metadata=edge.metadata,
                            )
                            self.edges[new_edge_key] = new_edge
                            self.relationship_index[
                                edge.relationship_type
                            ].add(new_edge_key)

                    # Remove old edge
                    del self.edges[edge_key]
                    if (
                        edge_key
                        in self.relationship_index[edge.relationship_type]
                    ):
                        self.relationship_index[edge.relationship_type].remove(
                            edge_key
                        )

            # Remove old node
            del self.nodes[node_id]
            if node_id in self.node_type_index[node.node_type]:
                self.node_type_index[node.node_type].remove(node_id)
            if node_id in self.node_embeddings:
                del self.node_embeddings[node_id]

        # Update primary node
        primary_node.content = merged_content
        primary_node.metadata = merged_metadata
        primary_node.updated_at = datetime.now()
        primary_node.confidence = min(1.0, primary_node.confidence + 0.2)

        logger.debug(
            "Merged %d knowledge nodes into %s", len(node_ids), primary_id
        )

    def _consolidation_worker(self):
        """Background worker for knowledge consolidation."""
        while self.processing_active:
            try:
                if self.consolidation_queue:
                    self.consolidate_knowledge()
                time.sleep(300)  # Run every 5 minutes
            except Exception as exc:
                logger.error("Consolidation worker error: %s", exc)
                time.sleep(60)

    def _decay_worker(self):
        """Background worker for knowledge decay."""
        while self.processing_active:
            try:
                self._apply_knowledge_decay()
                time.sleep(3600)  # Run every hour
            except Exception as exc:
                logger.error("Decay worker error: %s", exc)
                time.sleep(300)

    def _apply_knowledge_decay(self):
        """Apply temporal decay to knowledge relevance."""
        with self.lock:
            current_time = datetime.now()
            decayed_nodes = []

            for node_id, node in self.nodes.items():
                if node.last_accessed:
                    hours_since_access = (
                        current_time - node.last_accessed
                    ).total_seconds() / 3600
                    decay_factor = self.decay_factor ** (
                        hours_since_access / 24
                    )  # Daily decay

                    # Apply decay to confidence
                    old_confidence = node.confidence
                    node.confidence *= decay_factor

                    # Mark for removal if confidence too low
                    if node.confidence < 0.1:
                        decayed_nodes.append(node_id)

            # Remove decayed nodes
            for node_id in decayed_nodes:
                self._remove_node(node_id)

            if decayed_nodes:
                logger.debug(
                    "Removed %d decayed knowledge nodes", len(decayed_nodes)
                )

    def _remove_node(self, node_id: str):
        """Remove a node and its relationships."""
        if node_id not in self.nodes:
            return

        node = self.nodes[node_id]
        node_type = node.node_type

        # Remove from indexes
        if node_id in self.node_type_index[node_type]:
            self.node_type_index[node_type].remove(node_id)
        if node_id in self.node_embeddings:
            del self.node_embeddings[node_id]

        # Remove relationships
        edges_to_remove = []
        for edge_key in self.edges:
            if node_id in edge_key:
                edges_to_remove.append(edge_key)

        for edge_key in edges_to_remove:
            edge = self.edges[edge_key]
            del self.edges[edge_key]
            if edge_key in self.relationship_index[edge.relationship_type]:
                self.relationship_index[edge.relationship_type].remove(
                    edge_key
                )

        # Remove node
        del self.nodes[node_id]

    def _enforce_capacity_limits(self):
        """Enforce maximum capacity limits."""
        if len(self.nodes) > self.max_nodes:
            # Remove oldest, least accessed nodes
            nodes_by_access = sorted(
                self.nodes.items(),
                key=lambda x: (
                    x[1].last_accessed or x[1].created_at,
                    x[1].access_count,
                ),
            )

            nodes_to_remove = len(self.nodes) - self.max_nodes
            for node_id, _ in nodes_by_access[:nodes_to_remove]:
                self._remove_node(node_id)

            logger.debug(
                "Enforced capacity limit, removed %d nodes", nodes_to_remove
            )

    def get_knowledge_stats(self) -> Dict[str, Any]:
        """Get knowledge graph statistics."""
        with self.lock:
            node_types = defaultdict(int)
            relationship_types = defaultdict(int)

            for node in self.nodes.values():
                node_types[node.node_type] += 1

            for edge in self.edges.values():
                relationship_types[edge.relationship_type] += 1

            return {
                "total_nodes": len(self.nodes),
                "total_edges": len(self.edges),
                "node_types": dict(node_types),
                "relationship_types": dict(relationship_types),
                "nodes_with_embeddings": len(self.node_embeddings),
                "consolidation_queue_size": len(self.consolidation_queue),
                "avg_confidence": (
                    sum(n.confidence for n in self.nodes.values())
                    / len(self.nodes)
                    if self.nodes
                    else 0
                ),
                "avg_access_count": (
                    sum(n.access_count for n in self.nodes.values())
                    / len(self.nodes)
                    if self.nodes
                    else 0
                ),
            }

    def export_knowledge_graph(self) -> Dict[str, Any]:
        """Export the entire knowledge graph."""
        with self.lock:
            return {
                "nodes": [node.to_dict() for node in self.nodes.values()],
                "edges": [edge.to_dict() for edge in self.edges.values()],
                "metadata": {
                    "export_timestamp": datetime.now().isoformat(),
                    "total_nodes": len(self.nodes),
                    "total_edges": len(self.edges),
                    "node_types": list(self.node_type_index.keys()),
                    "relationship_types": list(self.relationship_index.keys()),
                },
            }

    def import_knowledge_graph(self, data: Dict[str, Any]):
        """Import knowledge graph from exported data."""
        with self.lock:
            # Clear existing data
            self.nodes.clear()
            self.edges.clear()
            self.node_embeddings.clear()
            self.node_type_index.clear()
            self.relationship_index.clear()

            # Import nodes
            for node_data in data.get("nodes", []):
                node = KnowledgeNode(
                    id=node_data["id"],
                    content=node_data["content"],
                    node_type=node_data["node_type"],
                    metadata=node_data["metadata"],
                    created_at=datetime.fromisoformat(node_data["created_at"]),
                    updated_at=datetime.fromisoformat(node_data["updated_at"]),
                    confidence=node_data["confidence"],
                    access_count=node_data["access_count"],
                    last_accessed=(
                        datetime.fromisoformat(node_data["last_accessed"])
                        if node_data.get("last_accessed")
                        else None
                    ),
                )
                self.nodes[node.id] = node
                self.node_type_index[node.node_type].add(node.id)

                # Note: Embeddings are not exported/imported for simplicity

            # Import edges
            for edge_data in data.get("edges", []):
                edge = KnowledgeEdge(
                    source_id=edge_data["source_id"],
                    target_id=edge_data["target_id"],
                    relationship_type=edge_data["relationship_type"],
                    weight=edge_data["weight"],
                    metadata=edge_data["metadata"],
                    created_at=datetime.fromisoformat(edge_data["created_at"]),
                    access_count=edge_data["access_count"],
                    last_accessed=(
                        datetime.fromisoformat(edge_data["last_accessed"])
                        if edge_data.get("last_accessed")
                        else None
                    ),
                )
                edge_key = (edge.source_id, edge.target_id)
                self.edges[edge_key] = edge
                self.relationship_index[edge.relationship_type].add(edge_key)

            logger.info(
                "Imported knowledge graph with %d nodes and %d edges",
                len(self.nodes),
                len(self.edges),
            )


# Register with provider registry
def create_knowledge_memory_graph(config_manager=None, **kwargs):
    """Factory function for KnowledgeMemoryGraph."""
    return KnowledgeMemoryGraph(config_manager=config_manager, **kwargs)


provider_registry.register_lazy(
    "knowledge",
    "memory_graph",
    "mia.adaptive_intelligence.knowledge_memory_graph",
    "create_knowledge_memory_graph",
    default=True,
)
