import networkx as nx
from chromadb import Client, Settings

class AgentMemory:
    def __init__(self):
        self.kg = nx.DiGraph()
        self.vector_db = Client(Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory="memory/"
        ))
        self.collection = self.vector_db.create_collection("episodic")
        
    def store_experience(self, text, embedding):
        """Store text with vector embedding"""
        self.collection.add(
            documents=[text],
            embeddings=[embedding],
            ids=[str(len(self.collection.get()['ids']))]
        )
    
    def retrieve_context(self, query_embedding, top_k=3):
        """Semantic similarity search"""
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        return results['documents'][0]

# Usage:
# memory = AgentMemory()
# memory.store_experience("User prefers dark mode", embedding_vector)