import logging
import uuid
from typing import Any, Dict, List, Optional

import networkx as nx

# Import custom exceptions and error handling
from ..exceptions import InitializationError, MemoryError, ValidationError

# Optional Chroma import with fallback
try:
    import chromadb
    from chromadb import PersistentClient
    CHROMADB_AVAILABLE = True
except ImportError:
    chromadb = None
    PersistentClient = None
    CHROMADB_AVAILABLE = False
    chromadb = None
    PersistentClient = None

logger = logging.getLogger(__name__)

class AgentMemory:
    """Hybrid memory that tracks a knowledge graph plus optional vector store."""

    def __init__(
        self,
        persist_directory: str = "memory/",
        enable_vector: bool = True,
        enable_graph: bool = True,
    ) -> None:
        self.persist_directory = persist_directory
        self.enable_vector = enable_vector and CHROMADB_AVAILABLE
        self.enable_graph = enable_graph
        self.kg = nx.DiGraph() if self.enable_graph else None
        self.vector_db = None
        self.collection = None

        if self.enable_vector:
            self._init_vector_db()
        
    def _init_vector_db(self) -> None:
        """Initialize vector database with comprehensive error handling."""
        if not self.enable_vector:
            return
        if not CHROMADB_AVAILABLE:
            logger.warning("ChromaDB not available - vector memory disabled")
            return
            
        try:
            # Validate persist directory
            if not self.persist_directory:
                raise InitializationError("Persist directory not specified", "MISSING_PERSIST_DIR")
            
            # Create directory if it doesn't exist
            import os
            os.makedirs(self.persist_directory, exist_ok=True)
            
            # Use new ChromaDB API
            try:
                if not CHROMADB_AVAILABLE or PersistentClient is None:
                    raise InitializationError("ChromaDB not available", "CHROMADB_NOT_AVAILABLE")
                self.vector_db = PersistentClient(path=self.persist_directory)
            except Exception as e:
                raise InitializationError(f"Failed to create ChromaDB client: {str(e)}", 
                                       "CHROMADB_CLIENT_FAILED")
            
            # Get or create collection
            try:
                self.collection = self.vector_db.get_collection("episodic")
                logger.info("Retrieved existing ChromaDB collection")
            except Exception:
                try:
                    self.collection = self.vector_db.create_collection("episodic")
                    logger.info("Created new ChromaDB collection")
                except Exception as e:
                    raise InitializationError(f"Failed to create ChromaDB collection: {str(e)}", 
                                           "COLLECTION_CREATE_FAILED")
                
        except InitializationError:
            raise
        except Exception as e:
            raise InitializationError(f"ChromaDB initialization failed: {str(e)}", 
                                   "CHROMADB_INIT_FAILED")
        
    def store_experience(
        self,
        text: str,
        embedding: List[float],
        doc_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Store text with vector embedding and optional metadata."""
        if not text:
            raise ValidationError("Empty text provided", "EMPTY_TEXT")
            
        if not isinstance(text, str):
            raise ValidationError("Text must be a string", "INVALID_TEXT_TYPE")
            
        if not embedding:
            raise ValidationError("Empty embedding provided", "EMPTY_EMBEDDING")
            
        if not isinstance(embedding, list):
            raise ValidationError("Embedding must be a list", "INVALID_EMBEDDING_TYPE")
        
        if not self.enable_vector or not self.collection:
            raise MemoryError("Vector database not available", "DB_NOT_AVAILABLE")
            
        try:
            if doc_id is None:
                # Generate a unique ID
                doc_id = f"doc_{uuid.uuid4().hex[:8]}"
                
            # Validate doc_id
            if not isinstance(doc_id, str) or not doc_id.strip():
                raise ValidationError("Invalid document ID", "INVALID_DOC_ID")
                
            self.collection.add(
                documents=[text],
                embeddings=[embedding],
                ids=[doc_id],
                metadatas=[metadata or {}],
            )

            logger.debug("Stored experience with ID: %s", doc_id)
            return {"id": doc_id, "text": text, "metadata": metadata or {}}
            
        except ValidationError:
            raise
        except Exception as e:
            raise MemoryError(f"Failed to store experience: {str(e)}", "STORE_FAILED")
    
    def retrieve_context(
        self, query_embedding: List[float], top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """Semantic similarity search returning structured entries."""
        if not query_embedding:
            raise ValidationError("Empty query embedding provided", "EMPTY_QUERY_EMBEDDING")
            
        if not isinstance(query_embedding, list):
            raise ValidationError("Query embedding must be a list", "INVALID_QUERY_EMBEDDING_TYPE")
            
        if not isinstance(top_k, int) or top_k <= 0:
            raise ValidationError("top_k must be a positive integer", "INVALID_TOP_K")
            
        if not self.enable_vector or not self.collection:
            logger.warning("Vector database not available")
            return []
            
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=min(top_k, 100),
            )

            if results is None:
                return []

            documents = results.get("documents", [[]]) or [[]]
            ids = results.get("ids", [[]]) or [[]]
            metadatas = results.get("metadatas", [[]]) or [[]]
            distances = results.get("distances", [[]]) or [[]]

            payload: List[Dict[str, Any]] = []
            for doc, doc_id, meta, distance in zip(
                documents[0],
                ids[0] if ids else [],
                metadatas[0] if metadatas else [],
                distances[0] if distances else [],
            ):
                payload.append(
                    {
                        "id": doc_id,
                        "text": doc,
                        "metadata": meta or {},
                        "distance": distance,
                    }
                )

            logger.debug("Retrieved %s context documents", len(payload))
            return payload
            
        except ValidationError:
            raise
        except Exception as e:
            raise MemoryError(f"Failed to retrieve context: {str(e)}", "RETRIEVE_FAILED")
    
    def get_all_documents(self) -> List[Dict[str, Any]]:
        """Get all stored documents from vector store."""
        if not self.enable_vector or not self.collection:
            return []
            
        try:
            results = self.collection.get()
            if results is None:
                return []
            docs = results.get("documents", []) or []
            ids = results.get("ids", []) or []
            metadatas = results.get("metadatas", []) or []
            payload: List[Dict[str, Any]] = []
            for doc, doc_id, meta in zip(docs, ids, metadatas):
                payload.append({"id": doc_id, "text": doc, "metadata": meta or {}})
            return payload
        except Exception as e:
            logger.error(f"Failed to get documents: {e}")
            return []

    # Knowledge graph helpers -------------------------------------------------

    def add_relation(
        self,
        source: str,
        relation: str,
        target: str,
        *,
        weight: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        if not self.enable_graph or self.kg is None:
            raise MemoryError("Knowledge graph not available", "GRAPH_NOT_AVAILABLE")
        if not source or not target:
            raise ValidationError("Source and target are required", "INVALID_KG_NODE")
        relation_label = relation or "related_to"
        self.kg.add_node(source)
        self.kg.add_node(target)
        self.kg.add_edge(
            source,
            target,
            relation=relation_label,
            metadata=metadata or {},
            weight=weight,
        )
        logger.debug("Added relation %s -[%s]-> %s", source, relation_label, target)
        return True

    def get_relations(self, node: str) -> List[Dict[str, Any]]:
        if not self.enable_graph or self.kg is None:
            return []
        if node not in self.kg:
            return []
        payload: List[Dict[str, Any]] = []
        for _, target, data in self.kg.out_edges(node, data=True):
            payload.append(
                {
                    "source": node,
                    "target": target,
                    "relation": data.get("relation"),
                    "metadata": data.get("metadata", {}),
                    "weight": data.get("weight"),
                }
            )
        return payload

    def find_related(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        if not self.enable_graph or self.kg is None:
            return []
        query_lower = query.lower()
        matches: List[Dict[str, Any]] = []

        for source, target, data in self.kg.edges(data=True):
            if (
                query_lower in str(source).lower()
                or query_lower in str(target).lower()
                or query_lower in str(data.get("relation", "")).lower()
            ):
                matches.append(
                    {
                        "source": source,
                        "target": target,
                        "relation": data.get("relation"),
                        "metadata": data.get("metadata", {}),
                        "weight": data.get("weight"),
                    }
                )
                if len(matches) >= limit:
                    break

        if len(matches) < limit:
            for node, data in self.kg.nodes(data=True):
                if query_lower in str(node).lower():
                    matches.append(
                        {
                            "source": node,
                            "target": None,
                            "relation": "node",
                            "metadata": data,
                            "weight": None,
                        }
                    )
                    if len(matches) >= limit:
                        break
        return matches

    def clear_graph(self) -> None:
        if self.enable_graph and self.kg is not None:
            self.kg.clear()
    
    def is_available(self) -> bool:
        """Check if vector database is available."""
        if self.enable_vector and self.collection is not None:
            return True
        if self.enable_graph and self.kg is not None:
            return True
        return False
    
    def get_status(self) -> dict:
        """Get status of memory components."""
        return {
            "knowledge_graph_available": self.enable_graph and self.kg is not None,
            "graph_nodes": self.kg.number_of_nodes() if self.kg else 0,
            "graph_edges": self.kg.number_of_edges() if self.kg else 0,
            "vector_db_available": self.enable_vector and self.collection is not None,
            "chromadb_available": CHROMADB_AVAILABLE,
            "persist_directory": self.persist_directory,
            "document_count": len(self.get_all_documents()) if self.enable_vector else 0,
        }

# Usage:
# memory = AgentMemory()
# memory.store_experience("User prefers dark mode", embedding_vector)