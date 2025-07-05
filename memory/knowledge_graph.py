import networkx as nx
import logging
from typing import Optional, List, Any

# Optional Chroma import with fallback
try:
    import chromadb
    from chromadb import PersistentClient
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    chromadb = None
    PersistentClient = None

logger = logging.getLogger(__name__)

class AgentMemory:
    def __init__(self, persist_directory="memory/"):
        self.kg = nx.DiGraph()
        self.vector_db = None
        self.collection = None
        self.persist_directory = persist_directory
        
        self._init_vector_db()
        
    def _init_vector_db(self):
        """Initialize vector database with new Chroma API."""
        if not CHROMADB_AVAILABLE:
            logger.warning("ChromaDB not available - vector memory disabled")
            return
            
        try:
            # Use new ChromaDB API
            self.vector_db = PersistentClient(path=self.persist_directory)
            
            # Get or create collection
            try:
                self.collection = self.vector_db.get_collection("episodic")
                logger.info("Retrieved existing ChromaDB collection")
            except:
                self.collection = self.vector_db.create_collection("episodic")
                logger.info("Created new ChromaDB collection")
                
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            self.vector_db = None
            self.collection = None
        
    def store_experience(self, text: str, embedding: List[float], doc_id: Optional[str] = None) -> bool:
        """Store text with vector embedding"""
        if not self.collection:
            logger.warning("Vector database not available")
            return False
            
        try:
            if doc_id is None:
                # Generate a simple ID
                doc_id = f"doc_{len(self.get_all_documents())}"
                
            self.collection.add(
                documents=[text],
                embeddings=[embedding],
                ids=[doc_id]
            )
            logger.debug(f"Stored experience with ID: {doc_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to store experience: {e}")
            return False
    
    def retrieve_context(self, query_embedding: List[float], top_k: int = 3) -> List[str]:
        """Semantic similarity search"""
        if not self.collection:
            logger.warning("Vector database not available")
            return []
            
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k
            )
            documents = results.get('documents', [[]])[0]
            logger.debug(f"Retrieved {len(documents)} context documents")
            return documents
        except Exception as e:
            logger.error(f"Failed to retrieve context: {e}")
            return []
    
    def get_all_documents(self) -> List[str]:
        """Get all stored documents."""
        if not self.collection:
            return []
            
        try:
            results = self.collection.get()
            return results.get('documents', [])
        except Exception as e:
            logger.error(f"Failed to get documents: {e}")
            return []
    
    def is_available(self) -> bool:
        """Check if vector database is available."""
        return self.collection is not None
    
    def get_status(self) -> dict:
        """Get status of memory components."""
        return {
            "knowledge_graph_available": True,
            "vector_db_available": self.collection is not None,
            "chromadb_available": CHROMADB_AVAILABLE,
            "persist_directory": self.persist_directory,
            "document_count": len(self.get_all_documents())
        }

# Usage:
# memory = AgentMemory()
# memory.store_experience("User prefers dark mode", embedding_vector)