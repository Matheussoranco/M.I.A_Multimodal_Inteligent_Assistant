import networkx as nx
import logging
from typing import Optional, List, Any

# Import custom exceptions and error handling
from ..exceptions import MemoryError, InitializationError, ValidationError
from ..error_handler import global_error_handler, with_error_handling, safe_execute

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
    def __init__(self, persist_directory="memory/"):
        self.kg = nx.DiGraph()
        self.vector_db = None
        self.collection = None
        self.persist_directory = persist_directory
        
        self._init_vector_db()
        
    def _init_vector_db(self):
        """Initialize vector database with comprehensive error handling."""
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
        
    def store_experience(self, text: str, embedding: List[float], doc_id: Optional[str] = None) -> bool:
        """Store text with vector embedding with comprehensive validation."""
        if not text:
            raise ValidationError("Empty text provided", "EMPTY_TEXT")
            
        if not isinstance(text, str):
            raise ValidationError("Text must be a string", "INVALID_TEXT_TYPE")
            
        if not embedding:
            raise ValidationError("Empty embedding provided", "EMPTY_EMBEDDING")
            
        if not isinstance(embedding, list):
            raise ValidationError("Embedding must be a list", "INVALID_EMBEDDING_TYPE")
        
        if not self.collection:
            raise MemoryError("Vector database not available", "DB_NOT_AVAILABLE")
            
        try:
            if doc_id is None:
                # Generate a unique ID
                import uuid
                doc_id = f"doc_{uuid.uuid4().hex[:8]}"
                
            # Validate doc_id
            if not isinstance(doc_id, str) or not doc_id.strip():
                raise ValidationError("Invalid document ID", "INVALID_DOC_ID")
                
            self.collection.add(
                documents=[text],
                embeddings=[embedding],
                ids=[doc_id]
            )
            
            logger.debug(f"Stored experience with ID: {doc_id}")
            return True
            
        except ValidationError:
            raise
        except Exception as e:
            raise MemoryError(f"Failed to store experience: {str(e)}", "STORE_FAILED")
    
    def retrieve_context(self, query_embedding: List[float], top_k: int = 3) -> List[str]:
        """Semantic similarity search with comprehensive validation."""
        if not query_embedding:
            raise ValidationError("Empty query embedding provided", "EMPTY_QUERY_EMBEDDING")
            
        if not isinstance(query_embedding, list):
            raise ValidationError("Query embedding must be a list", "INVALID_QUERY_EMBEDDING_TYPE")
            
        if not isinstance(top_k, int) or top_k <= 0:
            raise ValidationError("top_k must be a positive integer", "INVALID_TOP_K")
            
        if not self.collection:
            logger.warning("Vector database not available")
            return []
            
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=min(top_k, 100)  # Limit to reasonable number
            )
            
            if results is None:
                return []
                
            documents = results.get('documents', [[]])
            if documents and len(documents) > 0:
                documents = documents[0]
            else:
                documents = []
            
            # Validate results
            if not isinstance(documents, list):
                raise MemoryError("Invalid query results format", "INVALID_RESULTS")
                
            logger.debug(f"Retrieved {len(documents)} context documents")
            return documents
            
        except ValidationError:
            raise
        except Exception as e:
            raise MemoryError(f"Failed to retrieve context: {str(e)}", "RETRIEVE_FAILED")
    
    def get_all_documents(self) -> List[str]:
        """Get all stored documents."""
        if not self.collection:
            return []
            
        try:
            results = self.collection.get()
            if results is None:
                return []
            documents = results.get('documents', [])
            return documents if documents is not None else []
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