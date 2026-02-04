"""
Embedding Manager: Unified interface for multiple embedding model providers.
Supports OpenAI embeddings, HuggingFace/Sentence-Transformers, local models, and Ollama embeddings.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import numpy as np
import requests

from ..config_manager import ConfigManager
from ..error_handler import global_error_handler, with_error_handling
from ..exceptions import (
    ConfigurationError,
    InitializationError,
    LLMProviderError,
)

# Optional imports with error handling
try:
    from openai import OpenAI

    HAS_OPENAI = True
except ImportError:
    OpenAI = None  # type: ignore[misc, assignment]
    HAS_OPENAI = False

try:
    from sentence_transformers import SentenceTransformer  # type: ignore[import-unresolved]

    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    SentenceTransformer = None  # type: ignore[misc, assignment]
    HAS_SENTENCE_TRANSFORMERS = False

try:
    from transformers import AutoModel, AutoTokenizer
    import torch

    HAS_TRANSFORMERS = True
except ImportError:
    AutoModel = None  # type: ignore[misc, assignment]
    AutoTokenizer = None  # type: ignore[misc, assignment]
    torch = None  # type: ignore[misc, assignment]
    HAS_TRANSFORMERS = False

logger = logging.getLogger(__name__)


SUPPORTED_EMBEDDING_PROVIDERS = {
    "openai",
    "sentence-transformers",
    "huggingface",
    "local",
    "ollama",
    "cohere",
    "voyageai",
    "auto",
}

# Default models per provider
DEFAULT_EMBEDDING_MODELS = {
    "openai": "text-embedding-3-small",
    "sentence-transformers": "all-MiniLM-L6-v2",
    "huggingface": "sentence-transformers/all-MiniLM-L6-v2",
    "ollama": "nomic-embed-text",
    "cohere": "embed-english-v3.0",
    "voyageai": "voyage-2",
}


@dataclass
class EmbeddingConfig:
    """Configuration for embedding models."""
    
    provider: str = "auto"
    model_id: Optional[str] = None
    api_key: Optional[str] = None
    url: Optional[str] = None
    dimension: int = 768
    batch_size: int = 32
    normalize: bool = True
    cache_enabled: bool = True
    device: str = "auto"  # auto, cpu, cuda, mps
    max_length: int = 512
    pooling_strategy: str = "mean"  # mean, cls, max
    
    def validate(self) -> None:
        """Validate embedding configuration."""
        if self.provider not in SUPPORTED_EMBEDDING_PROVIDERS:
            raise ConfigurationError(
                f"Unsupported embedding provider: {self.provider}",
                "UNSUPPORTED_EMBEDDING_PROVIDER",
            )
        
        if self.dimension <= 0:
            raise ConfigurationError(
                "Embedding dimension must be positive",
                "INVALID_DIMENSION",
            )
        
        if self.batch_size <= 0:
            raise ConfigurationError(
                "Batch size must be positive",
                "INVALID_BATCH_SIZE",
            )


class EmbeddingManager:
    """Unified embedding manager supporting multiple providers."""

    @classmethod
    def detect_available_providers(cls) -> List[Dict[str, Any]]:
        """Detect available embedding providers."""
        providers = []
        
        # Check OpenAI
        if HAS_OPENAI and os.getenv("OPENAI_API_KEY"):
            providers.append({
                "name": "openai",
                "model": DEFAULT_EMBEDDING_MODELS["openai"],
                "type": "api",
                "available": True,
            })
        
        # Check Sentence Transformers (local)
        if HAS_SENTENCE_TRANSFORMERS:
            providers.append({
                "name": "sentence-transformers",
                "model": DEFAULT_EMBEDDING_MODELS["sentence-transformers"],
                "type": "local",
                "available": True,
            })
        
        # Check HuggingFace Transformers
        if HAS_TRANSFORMERS:
            providers.append({
                "name": "huggingface",
                "model": DEFAULT_EMBEDDING_MODELS["huggingface"],
                "type": "local",
                "available": True,
            })
        
        # Check Ollama
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            if response.status_code == 200:
                data = response.json()
                embedding_models = [
                    m for m in data.get("models", [])
                    if "embed" in m.get("name", "").lower() or "nomic" in m.get("name", "").lower()
                ]
                if embedding_models:
                    providers.append({
                        "name": "ollama",
                        "model": embedding_models[0]["name"],
                        "type": "local",
                        "available": True,
                        "models": [m["name"] for m in embedding_models],
                    })
        except Exception:
            pass
        
        # Check Cohere
        if os.getenv("COHERE_API_KEY"):
            providers.append({
                "name": "cohere",
                "model": DEFAULT_EMBEDDING_MODELS["cohere"],
                "type": "api",
                "available": True,
            })
        
        # Check VoyageAI
        if os.getenv("VOYAGE_API_KEY"):
            providers.append({
                "name": "voyageai",
                "model": DEFAULT_EMBEDDING_MODELS["voyageai"],
                "type": "api",
                "available": True,
            })
        
        return providers

    @classmethod
    def detect_ollama_embedding_models(cls) -> List[Dict[str, Any]]:
        """Detect Ollama embedding models."""
        models = []
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            if response.status_code == 200:
                data = response.json()
                for model in data.get("models", []):
                    name = model.get("name", "").lower()
                    # Filter for known embedding models
                    if any(kw in name for kw in ["embed", "nomic", "bge", "e5", "gte"]):
                        models.append({
                            "name": model.get("name"),
                            "size": model.get("size", 0),
                            "provider": "ollama",
                        })
        except Exception:
            pass
        return models

    @classmethod
    def interactive_model_selection(cls) -> Dict[str, Any]:
        """Interactive embedding model selection at startup."""
        print("\n" + "═" * 60)
        print("M.I.A - Embedding Model Selection")
        print("═" * 60)
        
        providers = cls.detect_available_providers()
        
        if not providers:
            print("\nNo embedding providers available.")
            print("Install sentence-transformers: pip install sentence-transformers")
            raise ConfigurationError(
                "No embedding providers available",
                "NO_EMBEDDING_PROVIDERS",
            )
        
        print("\nAvailable Embedding Providers:")
        for i, provider in enumerate(providers, 1):
            type_label = "LOCAL" if provider["type"] == "local" else "API"
            print(f"  {i}. [{type_label}] {provider['name'].upper()} ({provider['model']})")
        
        while True:
            try:
                choice = input(f"\nSelect provider [1-{len(providers)}] (default: 1): ").strip() or "1"
                idx = int(choice) - 1
                if 0 <= idx < len(providers):
                    selected = providers[idx]
                    break
                print("Invalid choice.")
            except (ValueError, KeyboardInterrupt, EOFError):
                idx = 0
                selected = providers[0]
                break

        print(f"\nSelected: {selected['name'].upper()} ({selected['model']})")

        return {
            "provider": selected["name"],
            "model_id": selected["model"],
            "type": selected["type"],
        }

    def __init__(
        self,
        provider: Optional[str] = None,
        model_id: Optional[str] = None,
        api_key: Optional[str] = None,
        url: Optional[str] = None,
        config_manager: Optional[ConfigManager] = None,
        auto_detect: bool = True,
        device: str = "auto",
        normalize: bool = True,
        cache_enabled: bool = True,
        **kwargs: Any,
    ) -> None:
        """Initialize the embedding manager.
        
        Args:
            provider: Embedding provider (openai, sentence-transformers, etc.)
            model_id: Model identifier
            api_key: API key for remote providers
            url: Custom URL endpoint
            config_manager: Optional ConfigManager instance
            auto_detect: Whether to auto-detect providers
            device: Device to use (auto, cpu, cuda, mps)
            normalize: Whether to normalize embeddings
            cache_enabled: Whether to cache embeddings
        """
        self.config_manager = config_manager or ConfigManager()
        
        # Auto-detect if needed
        import sys
        is_testing = "pytest" in sys.modules or os.getenv("TESTING") == "true"
        
        if auto_detect and not provider and not is_testing:
            try:
                detected = self.interactive_model_selection()
                provider = detected["provider"]
                model_id = detected["model_id"]
                logger.info(f"Auto-detected embedding provider: {provider}")
            except Exception as e:
                logger.warning(f"Auto-detection failed: {e}")
                # Default to sentence-transformers if available
                if HAS_SENTENCE_TRANSFORMERS:
                    provider = "sentence-transformers"
                    model_id = DEFAULT_EMBEDDING_MODELS["sentence-transformers"]
        
        self.provider = provider or "sentence-transformers"
        self.model_id = model_id or DEFAULT_EMBEDDING_MODELS.get(self.provider, "all-MiniLM-L6-v2")
        self.api_key = api_key
        self.url = url
        self.device = self._resolve_device(device)
        self.normalize = normalize
        self.cache_enabled = cache_enabled
        
        # Initialize model components
        self.client: Optional[Any] = None
        self.model: Optional[Any] = None
        self.tokenizer: Optional[Any] = None
        self._embedding_cache: Dict[str, np.ndarray] = {}
        self._dimension: Optional[int] = None
        self._available = True
        
        try:
            self._initialize_provider()
        except Exception as e:
            logger.warning(f"Failed to initialize embedding provider {self.provider}: {e}")
            self._available = False
            if is_testing:
                raise

    def _resolve_device(self, device: str) -> str:
        """Resolve the device to use for local models."""
        if device != "auto":
            return device
        
        if HAS_TRANSFORMERS and torch is not None:
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        return "cpu"

    def _initialize_provider(self) -> None:
        """Initialize the specific provider."""
        if self.provider == "openai":
            self._initialize_openai()
        elif self.provider == "sentence-transformers":
            self._initialize_sentence_transformers()
        elif self.provider in ("huggingface", "local"):
            self._initialize_huggingface()
        elif self.provider == "ollama":
            self._initialize_ollama()
        elif self.provider == "cohere":
            self._initialize_cohere()
        elif self.provider == "voyageai":
            self._initialize_voyageai()
        else:
            raise ConfigurationError(
                f"Unknown embedding provider: {self.provider}",
                "UNKNOWN_PROVIDER",
            )

    def _initialize_openai(self) -> None:
        """Initialize OpenAI embeddings."""
        if not HAS_OPENAI:
            raise InitializationError(
                "OpenAI package not installed. Run: pip install openai",
                "MISSING_DEPENDENCY",
            )
        
        api_key = self.api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ConfigurationError(
                "OpenAI API key not provided",
                "MISSING_API_KEY",
            )
        
        self.client = OpenAI(api_key=api_key)  # type: ignore[misc]
        self._dimension = 1536 if "3-large" in self.model_id else 1536 if "ada" in self.model_id else 1536
        
        # Dimension mapping for OpenAI models
        dimension_map = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
        }
        self._dimension = dimension_map.get(self.model_id, 1536)

    def _initialize_sentence_transformers(self) -> None:
        """Initialize Sentence Transformers."""
        if not HAS_SENTENCE_TRANSFORMERS:
            raise InitializationError(
                "sentence-transformers not installed. Run: pip install sentence-transformers",
                "MISSING_DEPENDENCY",
            )
        
        try:
            self.model = SentenceTransformer(self.model_id, device=self.device)  # type: ignore[misc]
            self._dimension = self.model.get_sentence_embedding_dimension()  # type: ignore[union-attr]
        except Exception as e:
            raise InitializationError(
                f"Failed to load Sentence Transformer model: {e}",
                "MODEL_LOAD_FAILED",
            )

    def _initialize_huggingface(self) -> None:
        """Initialize HuggingFace transformers for embeddings."""
        if not HAS_TRANSFORMERS:
            raise InitializationError(
                "transformers package not installed. Run: pip install transformers torch",
                "MISSING_DEPENDENCY",
            )
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)  # type: ignore[union-attr]
            self.model = AutoModel.from_pretrained(self.model_id)  # type: ignore[union-attr]
            
            if self.device == "cuda" and torch.cuda.is_available():  # type: ignore[union-attr]
                self.model = self.model.cuda()  # type: ignore[union-attr]
            elif self.device == "mps" and hasattr(torch.backends, "mps"):  # type: ignore[union-attr]
                self.model = self.model.to("mps")  # type: ignore[union-attr]
            
            self.model.eval()  # type: ignore[union-attr]
            self._dimension = self.model.config.hidden_size  # type: ignore[union-attr]
        except Exception as e:
            raise InitializationError(
                f"Failed to load HuggingFace model: {e}",
                "MODEL_LOAD_FAILED",
            )

    def _initialize_ollama(self) -> None:
        """Initialize Ollama embeddings."""
        self.url = self.url or os.getenv("OLLAMA_URL") or "http://localhost:11434"
        # Test connectivity
        try:
            response = requests.get(f"{self.url}/api/tags", timeout=5)
            if response.status_code != 200:
                raise InitializationError(
                    "Ollama server not responding",
                    "OLLAMA_NOT_AVAILABLE",
                )
        except requests.RequestException as e:
            raise InitializationError(
                f"Cannot connect to Ollama: {e}",
                "OLLAMA_CONNECTION_FAILED",
            )
        
        # Default dimension for common Ollama embedding models
        dimension_map = {
            "nomic-embed-text": 768,
            "mxbai-embed-large": 1024,
            "all-minilm": 384,
            "bge-m3": 1024,
        }
        self._dimension = dimension_map.get(self.model_id.split(":")[0], 768)

    def _initialize_cohere(self) -> None:
        """Initialize Cohere embeddings."""
        self.api_key = self.api_key or os.getenv("COHERE_API_KEY")
        if not self.api_key:
            raise ConfigurationError(
                "Cohere API key not provided",
                "MISSING_API_KEY",
            )
        self._dimension = 1024  # Default for embed-english-v3.0

    def _initialize_voyageai(self) -> None:
        """Initialize VoyageAI embeddings."""
        self.api_key = self.api_key or os.getenv("VOYAGE_API_KEY")
        if not self.api_key:
            raise ConfigurationError(
                "VoyageAI API key not provided",
                "MISSING_API_KEY",
            )
        self._dimension = 1024  # Default dimension

    @property
    def dimension(self) -> int:
        """Get the embedding dimension."""
        return self._dimension or 768

    @property
    def is_available(self) -> bool:
        """Check if the embedding manager is available."""
        return self._available

    def embed(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> np.ndarray:
        """Generate embeddings for text(s).
        
        Args:
            texts: Single text or list of texts to embed
            batch_size: Batch size for processing
            show_progress: Whether to show progress bar
            
        Returns:
            Numpy array of embeddings, shape (n_texts, dimension)
        """
        if isinstance(texts, str):
            texts = [texts]
        
        if not texts:
            return np.array([])
        
        # Check cache
        if self.cache_enabled:
            cached_results = []
            uncached_texts = []
            uncached_indices = []
            
            for i, text in enumerate(texts):
                cache_key = f"{self.provider}:{self.model_id}:{hash(text)}"
                if cache_key in self._embedding_cache:
                    cached_results.append((i, self._embedding_cache[cache_key]))
                else:
                    uncached_texts.append(text)
                    uncached_indices.append(i)
            
            if not uncached_texts:
                # All cached
                results = [None] * len(texts)
                for i, emb in cached_results:
                    results[i] = emb
                return np.array(results)
        else:
            uncached_texts = texts
            uncached_indices = list(range(len(texts)))
            cached_results = []
        
        # Generate embeddings for uncached texts
        if self.provider == "openai":
            embeddings = self._embed_openai(uncached_texts, batch_size)
        elif self.provider == "sentence-transformers":
            embeddings = self._embed_sentence_transformers(uncached_texts, batch_size, show_progress)
        elif self.provider in ("huggingface", "local"):
            embeddings = self._embed_huggingface(uncached_texts, batch_size)
        elif self.provider == "ollama":
            embeddings = self._embed_ollama(uncached_texts, batch_size)
        elif self.provider == "cohere":
            embeddings = self._embed_cohere(uncached_texts, batch_size)
        elif self.provider == "voyageai":
            embeddings = self._embed_voyageai(uncached_texts, batch_size)
        else:
            raise LLMProviderError(
                f"Embedding not implemented for provider: {self.provider}",
                "NOT_IMPLEMENTED",
            )
        
        # Normalize if requested
        if self.normalize:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)  # Avoid division by zero
            embeddings = embeddings / norms
        
        # Cache results
        if self.cache_enabled:
            for i, idx in enumerate(uncached_indices):
                cache_key = f"{self.provider}:{self.model_id}:{hash(uncached_texts[i])}"
                self._embedding_cache[cache_key] = embeddings[i]
        
        # Merge cached and new results
        if cached_results:
            results = [None] * len(texts)
            for i, emb in cached_results:
                results[i] = emb
            for i, idx in enumerate(uncached_indices):
                results[idx] = embeddings[i]
            return np.array(results)
        
        return embeddings

    def _embed_openai(self, texts: List[str], batch_size: int) -> np.ndarray:
        """Generate embeddings using OpenAI."""
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            response = self.client.embeddings.create(  # type: ignore[union-attr]
                model=self.model_id,
                input=batch,
            )
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)
        
        return np.array(all_embeddings)

    def _embed_sentence_transformers(
        self, texts: List[str], batch_size: int, show_progress: bool
    ) -> np.ndarray:
        """Generate embeddings using Sentence Transformers."""
        return self.model.encode(  # type: ignore[union-attr]
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=False,  # We normalize ourselves
        )

    def _embed_huggingface(self, texts: List[str], batch_size: int) -> np.ndarray:
        """Generate embeddings using HuggingFace transformers."""
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            # Tokenize
            inputs = self.tokenizer(  # type: ignore[misc]
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            
            # Move to device
            if self.device == "cuda":
                inputs = {k: v.cuda() for k, v in inputs.items()}
            elif self.device == "mps":
                inputs = {k: v.to("mps") for k, v in inputs.items()}
            
            # Get embeddings
            with torch.no_grad():  # type: ignore[union-attr]
                outputs = self.model(**inputs)  # type: ignore[misc]
                
            # Mean pooling
            attention_mask = inputs["attention_mask"]
            token_embeddings = outputs.last_hidden_state
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(  # type: ignore[union-attr]
                input_mask_expanded.sum(1), min=1e-9
            )
            
            all_embeddings.append(embeddings.cpu().numpy())
        
        return np.vstack(all_embeddings)

    def _embed_ollama(self, texts: List[str], batch_size: int) -> np.ndarray:
        """Generate embeddings using Ollama."""
        all_embeddings = []
        
        for text in texts:
            response = requests.post(
                f"{self.url}/api/embeddings",
                json={"model": self.model_id, "prompt": text},
                timeout=60,
            )
            response.raise_for_status()
            data = response.json()
            all_embeddings.append(data["embedding"])
        
        return np.array(all_embeddings)

    def _embed_cohere(self, texts: List[str], batch_size: int) -> np.ndarray:
        """Generate embeddings using Cohere."""
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            response = requests.post(
                "https://api.cohere.ai/v1/embed",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "texts": batch,
                    "model": self.model_id,
                    "input_type": "search_document",
                },
                timeout=60,
            )
            response.raise_for_status()
            data = response.json()
            all_embeddings.extend(data["embeddings"])
        
        return np.array(all_embeddings)

    def _embed_voyageai(self, texts: List[str], batch_size: int) -> np.ndarray:
        """Generate embeddings using VoyageAI."""
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            response = requests.post(
                "https://api.voyageai.com/v1/embeddings",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "input": batch,
                    "model": self.model_id,
                },
                timeout=60,
            )
            response.raise_for_status()
            data = response.json()
            all_embeddings.extend([item["embedding"] for item in data["data"]])
        
        return np.array(all_embeddings)

    def similarity(
        self,
        text1: Union[str, np.ndarray],
        text2: Union[str, np.ndarray],
    ) -> float:
        """Calculate cosine similarity between two texts or embeddings.
        
        Args:
            text1: First text or embedding
            text2: Second text or embedding
            
        Returns:
            Cosine similarity score between -1 and 1
        """
        # Get embeddings if strings
        if isinstance(text1, str):
            emb1 = self.embed(text1)[0]
        else:
            emb1 = text1
            
        if isinstance(text2, str):
            emb2 = self.embed(text2)[0]
        else:
            emb2 = text2
        
        # Calculate cosine similarity
        dot_product = np.dot(emb1, emb2)
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(dot_product / (norm1 * norm2))

    def find_most_similar(
        self,
        query: str,
        candidates: List[str],
        top_k: int = 5,
        threshold: float = 0.0,
    ) -> List[Tuple[int, str, float]]:
        """Find most similar texts to a query.
        
        Args:
            query: Query text
            candidates: List of candidate texts
            top_k: Number of top results to return
            threshold: Minimum similarity threshold
            
        Returns:
            List of (index, text, similarity_score) tuples
        """
        if not candidates:
            return []
        
        # Embed query and candidates
        query_embedding = self.embed(query)[0]
        candidate_embeddings = self.embed(candidates)
        
        # Calculate similarities
        similarities = np.dot(candidate_embeddings, query_embedding)
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            score = float(similarities[idx])
            if score >= threshold:
                results.append((int(idx), candidates[idx], score))
        
        return results

    def cluster_texts(
        self,
        texts: List[str],
        n_clusters: int = 5,
        method: str = "kmeans",
    ) -> Dict[str, Any]:
        """Cluster texts based on their embeddings.
        
        Args:
            texts: List of texts to cluster
            n_clusters: Number of clusters
            method: Clustering method (kmeans, hierarchical)
            
        Returns:
            Dictionary with cluster assignments and centroids
        """
        if len(texts) < n_clusters:
            n_clusters = len(texts)
        
        embeddings = self.embed(texts)
        
        try:
            from sklearn.cluster import KMeans, AgglomerativeClustering
        except ImportError:
            raise InitializationError(
                "scikit-learn required for clustering. Run: pip install scikit-learn",
                "MISSING_DEPENDENCY",
            )
        
        if method == "kmeans":
            clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = clusterer.fit_predict(embeddings)
            centroids = clusterer.cluster_centers_
        elif method == "hierarchical":
            clusterer = AgglomerativeClustering(n_clusters=n_clusters)
            labels = clusterer.fit_predict(embeddings)
            # Calculate centroids manually
            centroids = np.array([
                embeddings[labels == i].mean(axis=0)
                for i in range(n_clusters)
            ])
        else:
            raise ValueError(f"Unknown clustering method: {method}")
        
        # Group texts by cluster
        clusters = {}
        for i, label in enumerate(labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(texts[i])
        
        return {
            "labels": labels.tolist(),
            "centroids": centroids,
            "clusters": clusters,
            "n_clusters": n_clusters,
        }

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current embedding model."""
        return {
            "provider": self.provider,
            "model_id": self.model_id,
            "dimension": self.dimension,
            "device": self.device,
            "normalize": self.normalize,
            "cache_enabled": self.cache_enabled,
            "cache_size": len(self._embedding_cache),
            "available": self._available,
        }

    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        self._embedding_cache.clear()

    def save_embeddings(
        self,
        texts: List[str],
        embeddings: np.ndarray,
        filepath: str,
    ) -> None:
        """Save embeddings to a file.
        
        Args:
            texts: List of texts
            embeddings: Numpy array of embeddings
            filepath: Path to save file
        """
        data = {
            "texts": texts,
            "embeddings": embeddings.tolist(),
            "provider": self.provider,
            "model_id": self.model_id,
            "dimension": self.dimension,
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f)

    def load_embeddings(self, filepath: str) -> Tuple[List[str], np.ndarray]:
        """Load embeddings from a file.
        
        Args:
            filepath: Path to the saved file
            
        Returns:
            Tuple of (texts, embeddings)
        """
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        return data["texts"], np.array(data["embeddings"])
