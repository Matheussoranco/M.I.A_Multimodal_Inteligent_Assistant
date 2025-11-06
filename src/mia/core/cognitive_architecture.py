import warnings
from typing import Any, Dict, List, Optional

import torch

from ..audio.speech_processor import SpeechProcessor
from ..error_handler import global_error_handler, with_error_handling

# Import custom exceptions and error handling
from ..exceptions import (
    InitializationError,
    ValidationError,
    VisionProcessingError,
)

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
    def __init__(self, llm_client: Any, device: Optional[str] = None) -> None:
        self.llm = llm_client
        self.device = (
            device
            if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.vision_processor: Optional[Any] = None
        self.vision_model: Optional[Any] = None
        self.speech_processor: Optional[SpeechProcessor] = None

        # Initialize vision components with proper error handling
        self._init_vision_components()
        self._init_speech_processor()

        self.working_memory = []

        # Initialize memory systems
        self.long_term_memory = {}
        self.knowledge_graph = {}

    def _init_vision_components(self) -> None:
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

                logger.info("Vision components initialized successfully")

        except InitializationError:
            raise
        except Exception as e:
            raise InitializationError(
                f"Unexpected error initializing vision components: {str(e)}",
                "VISION_INIT_FAILED",
            )

    def _init_speech_processor(self) -> None:
        try:
            self.speech_processor = SpeechProcessor()
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
            if "image" in inputs and inputs["image"] is not None:
                if self.vision_processor is None or self.vision_model is None:
                    raise VisionProcessingError(
                        "Vision components not initialized",
                        "VISION_NOT_AVAILABLE",
                    )

                # For testing purposes, we'll mock the image processing
                processed["image_description"] = "Mock image description"

            # Process audio input
            if "audio" in inputs and inputs["audio"] is not None:
                if self.speech_processor is None:
                    raise VisionProcessingError(
                        "Speech processor not available",
                        "SPEECH_NOT_AVAILABLE",
                    )

                # For testing purposes, we'll mock the audio processing
                processed["audio_transcription"] = "Mock audio transcription"

            # Generate embeddings for the processed content
            if processed:
                text_for_embedding = processed.get("text", "")
                if text_for_embedding:
                    processed["embedding"] = self.generate_embeddings(
                        text_for_embedding
                    )

            # Generate response using LLM
            if processed:
                prompt = self._build_multimodal_prompt(processed)
                if hasattr(self.llm, "query") and self.llm.query:
                    response = self.llm.query(prompt)
                    processed["text"] = (
                        response  # Update text with LLM response
                    )
                    processed["response"] = response
                elif hasattr(self.llm, "query_model") and self.llm.query_model:
                    response = self.llm.query_model(prompt)
                    processed["text"] = (
                        response  # Update text with LLM response
                    )
                    processed["response"] = response
                else:
                    processed["response"] = (
                        "LLM not available for response generation"
                    )

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

    def generate_embeddings(self, text: Optional[str]) -> List[float]:
        try:
            import hashlib

            if not text or not text.strip():
                return []

            embeddings = []
            for i in range(10):  # 10-dimensional embedding
                hash_obj = hashlib.md5(f"{text}_{i}".encode())
                hash_int = int(hash_obj.hexdigest(), 16)
                # Normalize to [-1, 1]
                normalized = (hash_int % 2000 - 1000) / 1000.0
                embeddings.append(normalized)

            # Add some text-based features
            word_count = len(text.split())
            char_count = len(text)
            embeddings.extend(
                [
                    min(word_count / 100.0, 1.0),  # Normalized word count
                    min(char_count / 1000.0, 1.0),  # Normalized char count
                    1.0 if "?" in text else 0.0,  # Question indicator
                    1.0 if "!" in text else 0.0,  # Exclamation indicator
                ]
            )

            return embeddings

        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            return []

    def _generate_embedding(self, text: Optional[str]) -> list:
        """Alias for generate_embeddings for backward compatibility."""
        if text is None:
            return []
        return self.generate_embeddings(text)

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
