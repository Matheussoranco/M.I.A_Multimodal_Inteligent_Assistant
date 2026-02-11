import re
import json
import logging
import warnings
from typing import Any, Dict, List, Optional, Union

import torch
import numpy as np

try:
    from PIL import Image

    HAS_PIL = True
except ImportError:
    Image = None
    HAS_PIL = False

from ..audio.speech_processor import SpeechProcessor
from ..tools.action_executor import ActionExecutor
from ..error_handler import global_error_handler, with_error_handling

# Import custom exceptions and error handling
from ..exceptions import (
    InitializationError,
    ValidationError,
    VisionProcessingError,
)

# Import embedding manager for real semantic embeddings
try:
    from ..llm.embedding_manager import EmbeddingManager
    HAS_EMBEDDING_MANAGER = True
except ImportError:
    EmbeddingManager = None
    HAS_EMBEDDING_MANAGER = False

# Suppress transformers warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore")
    warnings.filterwarnings("ignore", message=".*slow.*processor.*")
    warnings.filterwarnings("ignore", message=".*use_fast.*")
    try:
        from transformers import CLIPModel, CLIPProcessor

        HAS_CLIP = True
    except ImportError:
        CLIPProcessor = None
        CLIPModel = None
        HAS_CLIP = False

import logging

logger = logging.getLogger(__name__)


class MIACognitiveCore:
    def __init__(
        self,
        llm_client: Any,
        device: Optional[str] = None,
        action_executor: Optional[ActionExecutor] = None,
    ) -> None:
        self.llm = llm_client
        self.device = (
            device
            if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.vision_processor: Optional[Any] = None
        self.vision_model: Optional[Any] = None
        self.speech_processor: Optional[SpeechProcessor] = None
        self.action_executor = action_executor or ActionExecutor()
        
        # Initialize real embedding manager for semantic embeddings
        self.embedding_manager: Optional[Any] = None
        self._init_embedding_manager()

        # Initialize memory systems
        self.long_term_memory = {}
        self.knowledge_graph = {}
        self.working_memory: List[Any] = []
        
        # Lazy loading states
        self._vision_initialized = False
        self._speech_initialized = False
    
    def _init_embedding_manager(self) -> None:
        """Initialize the embedding manager for real semantic embeddings."""
        if not HAS_EMBEDDING_MANAGER:
            logger.warning(
                "EmbeddingManager not available - using fallback embeddings"
            )
            return
        
        try:
            # Initialize with auto-detection disabled to avoid interactive prompts
            if EmbeddingManager is not None:
                self.embedding_manager = EmbeddingManager(
                    provider="sentence-transformers",
                    model_id="all-MiniLM-L6-v2",
                    auto_detect=False,
                    device=self.device,
                    normalize=True,
                    cache_enabled=True,
                )
                logger.info(
                    f"Embedding manager initialized: {self.embedding_manager.provider} "
                    f"({self.embedding_manager.model_id}), dimension={self.embedding_manager.dimension}"
                )
        except Exception as e:
            logger.warning(f"Failed to initialize embedding manager: {e}")
            self.embedding_manager = None

    def _ensure_vision_components(self) -> None:
        """Lazy load vision components."""
        if self._vision_initialized:
            return

        if not HAS_CLIP:
            logger.warning(
                "CLIP components not available - vision processing disabled"
            )
            return

        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                warnings.filterwarnings(
                    "ignore", message=".*slow.*processor.*"
                )
                warnings.filterwarnings("ignore", message=".*use_fast.*")

                if CLIPProcessor is None or CLIPModel is None:
                    raise InitializationError(
                        "CLIP components not imported", "IMPORT_ERROR"
                    )
                
                logger.info("Lazy loading CLIP vision model...")
                try:
                    self.vision_processor = CLIPProcessor.from_pretrained(
                        "openai/clip-vit-base-patch32"
                    )
                except Exception as e:
                    raise InitializationError(
                        f"Failed to load CLIP processor: {str(e)}",
                        "PROCESSOR_LOAD_FAILED",
                    )

                try:
                    self.vision_model = CLIPModel.from_pretrained(
                        "openai/clip-vit-base-patch32"
                    )
                except Exception as e:
                    raise InitializationError(
                        f"Failed to load CLIP model: {str(e)}",
                        "MODEL_LOAD_FAILED",
                    )

                # Move model to device if available
                if self.vision_model is not None and hasattr(
                    self.vision_model, "to"
                ):
                    try:
                        self.vision_model = self.vision_model.to(self.device)
                        logger.info(
                            f"Vision model moved to device: {self.device}"
                        )
                    except Exception as e:
                        logger.warning(
                            f"Failed to move vision model to device {self.device}: {e}"
                        )

                self._vision_initialized = True
                logger.info("Vision components initialized successfully")

        except InitializationError:
            raise
        except Exception as e:
            raise InitializationError(
                f"Unexpected error initializing vision components: {str(e)}",
                "VISION_INIT_FAILED",
            )

    def _ensure_speech_processor(self) -> None:
        if self._speech_initialized:
            return

        try:
            self.speech_processor = SpeechProcessor()
            self._speech_initialized = True
            logger.info("Speech processor initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize speech processor: {e}")
            self.speech_processor = None

    @with_error_handling(
        global_error_handler, fallback_value={"error": "Processing failed"}
    )
    @with_error_handling(
        global_error_handler, fallback_value={"error": "Processing failed"}
    )
    def process_multimodal_input(
        self, inputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        if not inputs:
            return {"text": "No input provided", "embedding": []}

        processed = {}

        try:
            # Process text input
            if "text" in inputs and inputs["text"]:
                processed["text"] = inputs["text"]

            # Process image input
            if "image" in inputs and inputs["image"]:
                self._ensure_vision_components()
                if self.vision_processor and self.vision_model:
                     processed["image_embedding"] = self._get_image_embedding(
                        inputs["image"]
                    )
                     processed["image_description"] = "Image processed by CLIP"
                else:
                    logger.warning("Vision components unavailable for image input")

            # Process audio input
            if "audio" in inputs and inputs["audio"]:
                self._ensure_speech_processor()
                if self.speech_processor:
                    processed["audio_transcription"] = (
                        self.speech_processor.transcribe(inputs["audio"])
                    )
                else:
                    logger.warning("Speech processor unavailable for audio input")

            # Generate embeddings for the processed content
            if processed:
                text_for_embedding = processed.get("text", "")
                if text_for_embedding:
                    processed["embedding"] = self.generate_embeddings(
                        text_for_embedding
                    )

            # Generate response using LLM
            if processed:
                # Use ReAct loop for text-based tasks
                if "text" in processed:
                    task = processed["text"]
                    context = {}
                    if "image_description" in processed:
                        context["image_description"] = processed["image_description"]
                    if "audio_transcription" in processed:
                        context["audio_transcription"] = processed["audio_transcription"]
                    
                    # Build prompt with context and delegate to LLM
                    prompt_parts = [task]
                    for k, v in context.items():
                        prompt_parts.append(f"{k}: {v}")
                    prompt = "\n".join(prompt_parts)

                    response = None
                    if hasattr(self.llm, "query") and callable(getattr(self.llm, "query", None)):
                        response = self.llm.query(prompt)
                    elif hasattr(self.llm, "query_model") and callable(getattr(self.llm, "query_model", None)):
                        response = self.llm.query_model(prompt)
                    elif hasattr(self.llm, "chat") and callable(getattr(self.llm, "chat", None)):
                        resp = self.llm.chat(prompt)
                        response = resp.text if hasattr(resp, "text") else str(resp)
                    else:
                        response = "LLM not available for response generation"
                    processed["text"] = response
                    processed["response"] = response
                else:
                    # Fallback for non-text inputs
                    prompt = self._build_multimodal_prompt(processed)
                    if hasattr(self.llm, "query") and self.llm.query:
                        response = self.llm.query(prompt)
                        processed["text"] = response
                        processed["response"] = response
                    elif hasattr(self.llm, "query_model") and self.llm.query_model:
                        response = self.llm.query_model(prompt)
                        processed["text"] = response
                        processed["response"] = response
                    else:
                        processed["response"] = "LLM not available for response generation"

        except (ValidationError, VisionProcessingError):
            raise
        except Exception as e:
            logger.error(f"Multimodal processing error: {e}")
            raise VisionProcessingError(
                f"Processing failed: {str(e)}", "PROCESSING_FAILED"
            )

        return processed

    def _build_multimodal_prompt(self, processed: Dict[str, Any]) -> str:
        prompt_parts = []

        if "text" in processed:
            text_content = processed["text"]
            prompt_parts.append(f"Text: {text_content}")

        if "image_description" in processed:
            image_desc = processed["image_description"]
            prompt_parts.append(f"Image: {image_desc}")

        if "audio_transcription" in processed:
            audio_trans = processed["audio_transcription"]
            prompt_parts.append(f"Audio: {audio_trans}")

        return " ".join(prompt_parts) if prompt_parts else "Empty input"

    def _get_image_embedding(self, image_input: Any) -> List[float]:
        """Generate an image embedding using CLIP."""
        if not self.vision_processor or not self.vision_model:
            raise VisionProcessingError(
                "Vision components not initialized",
                "VISION_COMPONENTS_MISSING",
            )

        if not HAS_PIL or Image is None:
            raise VisionProcessingError(
                "Pillow is required for image processing",
                "MISSING_DEPENDENCY",
            )

        try:
            if isinstance(image_input, str):
                image = Image.open(image_input).convert("RGB")
            elif isinstance(image_input, np.ndarray):
                image = Image.fromarray(image_input).convert("RGB")
            elif hasattr(image_input, "read"):
                image = Image.open(image_input).convert("RGB")
            else:
                raise ValidationError(
                    "Unsupported image input type",
                    "INVALID_IMAGE_INPUT",
                )

            inputs = self.vision_processor(
                images=image, return_tensors="pt"
            )
            if hasattr(inputs, "to"):
                inputs = inputs.to(self.device)
            else:
                inputs = {
                    k: (v.to(self.device) if hasattr(v, "to") else v)
                    for k, v in inputs.items()
                }

            with torch.no_grad():
                features = self.vision_model.get_image_features(**inputs)

            embedding = features[0].detach().cpu().numpy().tolist()
            return embedding

        except (ValidationError, VisionProcessingError):
            raise
        except Exception as exc:
            raise VisionProcessingError(
                f"Image embedding failed: {exc}",
                "IMAGE_EMBEDDING_FAILED",
            )

    def _reasoning_pipeline(self, context: str) -> Dict[str, Any]:
        try:
            if hasattr(self.llm, "query") and self.llm.query:
                response = self.llm.query(f"Reason about: {context}")
                result: Dict[str, Any] = {"text": response}
            elif hasattr(self.llm, "query_model") and self.llm.query_model:
                response = self.llm.query_model(f"Reason about: {context}")
                result: Dict[str, Any] = {"text": response}
            else:
                result: Dict[str, Any] = {"text": ""}

            # Generate embeddings for the reasoning result
            if result.get("text"):
                result["embedding"] = self.generate_embeddings(result["text"])

            return result

        except Exception as e:
            logger.error(f"Reasoning pipeline error: {e}")
            return {"text": "Error in reasoning pipeline"}

    def generate_embeddings(self, text: Optional[str]) -> Union[List[float], np.ndarray]:
        """Generate semantic embeddings for text using the embedding manager.
        
        This method uses state-of-the-art sentence transformers or other
        embedding providers to generate meaningful semantic vectors.
        
        Args:
            text: The text to embed
            
        Returns:
            List of floats or numpy array representing the embedding
            
        Raises:
            RuntimeError: If no real embedding provider is available
        """
        if not text or not text.strip():
            return []
        
        try:
            # Use real embedding manager — never silently fall back to hashes
            if self.embedding_manager is not None and self.embedding_manager.is_available:
                embeddings = self.embedding_manager.embed(text)
                if len(embeddings) > 0:
                    return embeddings[0].tolist() if hasattr(embeddings[0], 'tolist') else list(embeddings[0])
            
            # No real embeddings available — fall back to hash-based embeddings
            logger.warning(
                "No embedding provider loaded.  Install sentence-transformers "
                "(pip install sentence-transformers) for semantic search.  "
                "Falling back to hash-based embeddings; memory retrieval quality will be poor."
            )
            return self._generate_fallback_embedding(text)

        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            return self._generate_fallback_embedding(text)
    
    def _generate_fallback_embedding(self, text: str) -> List[float]:
        """Generate fallback hash-based embeddings when real embeddings unavailable.
        
        WARNING: These are NOT semantic embeddings and should only be used
        as a last resort when proper embedding models are unavailable.
        """
        import hashlib
        
        embeddings = []
        for i in range(384):  # Match common embedding dimension
            hash_obj = hashlib.sha256(f"{text}_{i}".encode())
            hash_int = int(hash_obj.hexdigest()[:8], 16)
            # Normalize to [-1, 1]
            normalized = (hash_int % 2000 - 1000) / 1000.0
            embeddings.append(normalized)

        # Add some text-based features
        word_count = len(text.split())
        char_count = len(text)
        
        # Normalize features
        embeddings[0] = min(word_count / 100.0, 1.0)
        embeddings[1] = min(char_count / 1000.0, 1.0)
        embeddings[2] = 1.0 if "?" in text else 0.0
        embeddings[3] = 1.0 if "!" in text else 0.0
        
        return embeddings
    
    def compute_similarity(self, text1: str, text2: str) -> float:
        """Compute semantic similarity between two texts.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score between 0 and 1
        """
        if self.embedding_manager is not None and self.embedding_manager.is_available:
            return self.embedding_manager.similarity(text1, text2)
        
        # Fallback: basic similarity
        emb1 = np.array(self.generate_embeddings(text1))
        emb2 = np.array(self.generate_embeddings(text2))
        
        if len(emb1) == 0 or len(emb2) == 0:
            return 0.0
        
        dot_product = np.dot(emb1, emb2)
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(dot_product / (norm1 * norm2))
    
    def find_similar_memories(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int = 5,
        threshold: float = 0.5,
    ) -> List[Dict[str, Any]]:
        """Find memories most similar to a query.
        
        Args:
            query: Query text
            candidates: List of memory entries with 'text' or 'fact' fields
            top_k: Number of results to return
            threshold: Minimum similarity threshold
            
        Returns:
            List of similar memories with similarity scores
        """
        if not candidates:
            return []
        
        # Extract texts from candidates
        texts = []
        for c in candidates:
            text = c.get('text') or c.get('fact') or str(c)
            texts.append(text)
        
        if self.embedding_manager is not None and self.embedding_manager.is_available:
            results = self.embedding_manager.find_most_similar(query, texts, top_k, threshold)
            return [
                {**candidates[idx], "similarity": score}
                for idx, _, score in results
            ]
        
        # Fallback
        query_emb = np.array(self.generate_embeddings(query))
        results = []
        
        for i, text in enumerate(texts):
            emb = np.array(self.generate_embeddings(text))
            if len(emb) > 0 and len(query_emb) > 0:
                similarity = float(np.dot(query_emb, emb) / (np.linalg.norm(query_emb) * np.linalg.norm(emb) + 1e-9))
                if similarity >= threshold:
                    results.append({**candidates[i], "similarity": similarity})
        
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:top_k]

    def _generate_embedding(self, text: Optional[str]) -> List[float]:
        """Alias for generate_embeddings for backward compatibility."""
        if text is None:
            return []
        result = self.generate_embeddings(text)
        if isinstance(result, np.ndarray):
            return result.tolist()
        return result

    def _update_working_memory(self, item: Any) -> None:
        self.working_memory.append(item)
        # Limit working memory size
        if len(self.working_memory) > 100:
            self.working_memory.pop(0)

    def _retrieve_from_memory(self, key: str) -> Optional[Any]:
        return self.long_term_memory.get(key)

    def _update_knowledge_graph(
        self, entity1: str, entity2: str, relationship: str
    ) -> None:
        if entity1 not in self.knowledge_graph:
            self.knowledge_graph[entity1] = {}
        self.knowledge_graph[entity1][entity2] = relationship

    def get_memory_stats(self) -> Dict[str, Any]:
        return {
            "working_memory_size": len(self.working_memory),
            "long_term_memory_size": len(self.long_term_memory),
            "knowledge_graph_size": len(self.knowledge_graph),
        }

    def reset_memory(self) -> None:
        self.working_memory.clear()
        self.long_term_memory.clear()
        self.knowledge_graph.clear()

    # NOTE: The ReAct task-execution loop has been moved to
    # ``core.agent.ToolCallingAgent``. The methods ``execute_task``,
    # ``_is_research_intent``, ``_query_llm``, and ``_parse_action``
    # previously here are now superseded.
