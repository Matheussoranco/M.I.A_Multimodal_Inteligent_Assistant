"""
Multimodal Perception Suite
"""

import asyncio
import base64
import io
import json
import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from ..providers import provider_registry

logger = logging.getLogger(__name__)


class ModalityType(Enum):
    """Supported perception modalities."""

    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    DOCUMENT = "document"
    SENSOR = "sensor"
    MULTIMODAL = "multimodal"


class ProcessingStatus(Enum):
    """Processing status for perception tasks."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class PerceptionInput:
    """Input data for perception processing."""

    modality: ModalityType
    data: Any  # Raw data (bytes, string, etc.)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    source: str = "unknown"
    format: Optional[str] = None  # file extension, mime type, etc.


@dataclass
class PerceptionResult:
    """Result of perception processing."""

    input_id: str
    modality: ModalityType
    status: ProcessingStatus
    extracted_data: Dict[str, Any] = field(default_factory=dict)
    confidence_scores: Dict[str, float] = field(default_factory=dict)
    processing_time: float = 0.0
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class MultimodalContext:
    """Context for multimodal fusion."""

    inputs: List[PerceptionResult] = field(default_factory=list)
    fused_insights: Dict[str, Any] = field(default_factory=dict)
    cross_modal_relations: List[Dict[str, Any]] = field(default_factory=list)
    confidence: float = 0.0
    processing_metadata: Dict[str, Any] = field(default_factory=dict)


class PerceptionProcessor:
    """
    Base class for modality-specific perception processors.

    Each processor handles a specific modality and extracts relevant information.
    """

    def __init__(self, modality: ModalityType, config: Optional[Dict[str, Any]] = None):
        self.modality = modality
        self.config = config or {}
        self.is_available = False
        self.capabilities = []

    def check_availability(self) -> bool:
        """Check if this processor is available for use."""
        return self.is_available

    async def process(self, input_data: PerceptionInput) -> PerceptionResult:
        """Process input data and return perception results."""
        raise NotImplementedError("Subclasses must implement process method")

    def get_capabilities(self) -> List[str]:
        """Get list of processing capabilities."""
        return self.capabilities.copy()


class TextProcessor(PerceptionProcessor):
    """Text processing and analysis."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(ModalityType.TEXT, config)
        self.is_available = True
        self.capabilities = [
            "sentiment_analysis",
            "entity_extraction",
            "language_detection",
            "keyword_extraction",
            "text_classification",
            "summarization",
        ]

    async def process(self, input_data: PerceptionInput) -> PerceptionResult:
        """Process text input."""
        start_time = datetime.now()

        try:
            text_content = self._extract_text(input_data.data)

            # Perform various text analyses
            results = {
                "text_length": len(text_content),
                "word_count": len(text_content.split()),
                "language": self._detect_language(text_content),
                "sentiment": self._analyze_sentiment(text_content),
                "entities": self._extract_entities(text_content),
                "keywords": self._extract_keywords(text_content),
                "summary": self._generate_summary(text_content),
            }

            processing_time = (datetime.now() - start_time).total_seconds()

            return PerceptionResult(
                input_id=f"{input_data.modality.value}_{hash(text_content) % 10000}",
                modality=input_data.modality,
                status=ProcessingStatus.COMPLETED,
                extracted_data=results,
                confidence_scores={
                    "overall": 0.85,
                    "sentiment": 0.8,
                    "entities": 0.75,
                    "keywords": 0.7,
                },
                processing_time=processing_time,
                metadata=input_data.metadata,
            )

        except Exception as exc:
            processing_time = (datetime.now() - start_time).total_seconds()
            return PerceptionResult(
                input_id=f"{input_data.modality.value}_error",
                modality=input_data.modality,
                status=ProcessingStatus.FAILED,
                error_message=str(exc),
                processing_time=processing_time,
            )

    def _extract_text(self, data: Any) -> str:
        """Extract text content from various formats."""
        if isinstance(data, str):
            return data
        elif isinstance(data, bytes):
            return data.decode("utf-8", errors="ignore")
        elif isinstance(data, dict) and "text" in data:
            return str(data["text"])
        else:
            return str(data)

    def _detect_language(self, text: str) -> str:
        """Simple language detection."""
        # Basic heuristics - in real implementation would use langdetect or similar
        if any(word in text.lower() for word in ["the", "and", "is"]):
            return "en"
        elif any(word in text.lower() for word in ["el", "la", "los"]):
            return "es"
        elif any(word in text.lower() for word in ["der", "die", "das"]):
            return "de"
        else:
            return "unknown"

    def _analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Simple sentiment analysis."""
        positive_words = [
            "good",
            "great",
            "excellent",
            "amazing",
            "wonderful",
            "fantastic",
        ]
        negative_words = ["bad", "terrible", "awful", "horrible", "worst", "hate"]

        words = text.lower().split()
        positive_count = sum(1 for word in words if word in positive_words)
        negative_count = sum(1 for word in words if word in negative_words)

        total_sentiment_words = positive_count + negative_count
        if total_sentiment_words == 0:
            sentiment = "neutral"
            score = 0.5
        else:
            score = positive_count / total_sentiment_words
            if score > 0.6:
                sentiment = "positive"
            elif score < 0.4:
                sentiment = "negative"
            else:
                sentiment = "neutral"

        return {
            "sentiment": sentiment,
            "score": score,
            "positive_words": positive_count,
            "negative_words": negative_count,
        }

    def _extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """Simple entity extraction."""
        # Basic pattern matching - in real implementation would use NER models
        entities = []

        # Simple email extraction
        import re

        emails = re.findall(
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", text
        )
        for email in emails:
            entities.append({"text": email, "type": "email", "confidence": 0.9})

        # Simple URL extraction
        urls = re.findall(
            r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+",
            text,
        )
        for url in urls:
            entities.append({"text": url, "type": "url", "confidence": 0.9})

        return entities

    def _extract_keywords(self, text: str) -> List[str]:
        """Simple keyword extraction."""
        words = text.lower().split()
        # Remove common stop words
        stop_words = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
            "is",
            "are",
            "was",
            "were",
        }
        keywords = [word for word in words if word not in stop_words and len(word) > 3]

        # Return most common keywords
        from collections import Counter

        return [word for word, _ in Counter(keywords).most_common(10)]

    def _generate_summary(self, text: str) -> str:
        """Simple text summarization."""
        sentences = text.split(".")
        if len(sentences) <= 2:
            return text

        # Return first and last sentences as simple summary
        return ". ".join([sentences[0], sentences[-1]]).strip()


class ImageProcessor(PerceptionProcessor):
    """Image processing and computer vision."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(ModalityType.IMAGE, config)
        self._check_cv2_availability()
        self.capabilities = [
            "object_detection",
            "face_recognition",
            "text_recognition",
            "color_analysis",
            "image_classification",
            "scene_description",
        ]

    def _check_cv2_availability(self):
        """Check if OpenCV is available."""
        try:
            import cv2

            self.cv2 = cv2
            self.is_available = True
        except ImportError:
            self.cv2 = None
            self.is_available = False
            logger.warning("OpenCV not available, image processing disabled")

    async def process(self, input_data: PerceptionInput) -> PerceptionResult:
        """Process image input."""
        if not self.is_available:
            return PerceptionResult(
                input_id=f"{input_data.modality.value}_unavailable",
                modality=input_data.modality,
                status=ProcessingStatus.FAILED,
                error_message="OpenCV not available",
            )

        start_time = datetime.now()

        try:
            # Decode image
            image = self._decode_image(input_data.data)
            if image is None:
                raise ValueError("Could not decode image")

            # Perform image analyses
            results = {
                "dimensions": image.shape[:2] if len(image.shape) >= 2 else (0, 0),
                "channels": image.shape[2] if len(image.shape) >= 3 else 1,
                "objects": self._detect_objects(image),
                "text": self._extract_text_from_image(image),
                "colors": self._analyze_colors(image),
                "description": self._describe_scene(image),
            }

            processing_time = (datetime.now() - start_time).total_seconds()

            return PerceptionResult(
                input_id=f"{input_data.modality.value}_{hash(str(input_data.data)) % 10000}",
                modality=input_data.modality,
                status=ProcessingStatus.COMPLETED,
                extracted_data=results,
                confidence_scores={
                    "overall": 0.75,
                    "objects": 0.7,
                    "text": 0.6,
                    "colors": 0.8,
                },
                processing_time=processing_time,
                metadata=input_data.metadata,
            )

        except Exception as exc:
            processing_time = (datetime.now() - start_time).total_seconds()
            return PerceptionResult(
                input_id=f"{input_data.modality.value}_error",
                modality=input_data.modality,
                status=ProcessingStatus.FAILED,
                error_message=str(exc),
                processing_time=processing_time,
            )

    def _decode_image(self, data: Any):
        """Decode image from various formats."""
        if self.cv2 is None:
            return None

        try:
            if isinstance(data, bytes):
                # Convert bytes to numpy array
                try:
                    import numpy as np

                    nparr = np.frombuffer(data, np.uint8)
                    return self.cv2.imdecode(nparr, self.cv2.IMREAD_COLOR)
                except ImportError:
                    return None
            elif isinstance(data, str):
                # Assume base64 encoded
                import base64

                try:
                    import numpy as np

                    image_data = base64.b64decode(data)
                    nparr = np.frombuffer(image_data, np.uint8)
                    return self.cv2.imdecode(nparr, self.cv2.IMREAD_COLOR)
                except ImportError:
                    return None
            else:
                return None
        except Exception:
            return None

    def _detect_objects(self, image) -> List[Dict[str, Any]]:
        """Simple object detection (placeholder)."""
        # In real implementation, would use YOLO, SSD, or similar
        return [
            {
                "label": "unknown_object",
                "confidence": 0.5,
                "bbox": [0, 0, image.shape[1], image.shape[0]],
            }
        ]

    def _extract_text_from_image(self, image) -> str:
        """Extract text from image (placeholder)."""
        # In real implementation, would use Tesseract OCR or similar
        return ""

    def _analyze_colors(self, image) -> Dict[str, Any]:
        """Analyze dominant colors in image."""
        if self.cv2 is None:
            return {"colors": "opencv_unavailable"}

        try:
            # Convert to RGB if needed
            if len(image.shape) == 3:
                # Simple color histogram
                hist = self.cv2.calcHist(
                    [image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256]
                )
                return {
                    "dominant_colors": hist.flatten().argsort()[-3:][::-1].tolist(),
                    "color_distribution": "analyzed",
                }
            else:
                return {"colors": "grayscale_image"}
        except Exception:
            return {"colors": "analysis_failed"}

    def _describe_scene(self, image) -> str:
        """Generate scene description (placeholder)."""
        # In real implementation, would use CLIP or similar vision-language models
        return "An image with visual content"


class AudioProcessor(PerceptionProcessor):
    """Audio processing and speech analysis."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(ModalityType.AUDIO, config)
        self._check_audio_availability()
        self.capabilities = [
            "speech_to_text",
            "speaker_identification",
            "emotion_detection",
            "sound_classification",
            "audio_quality_analysis",
            "language_identification",
        ]

    def _check_audio_availability(self):
        """Check if audio processing libraries are available."""
        try:
            import speech_recognition as sr

            self.speech_recognition = sr
            self.is_available = True
        except ImportError:
            self.speech_recognition = None
            self.is_available = False
            logger.warning("Speech recognition not available, audio processing limited")

    async def process(self, input_data: PerceptionInput) -> PerceptionResult:
        """Process audio input."""
        start_time = datetime.now()

        try:
            # Process audio data
            results = {
                "duration": self._get_audio_duration(input_data.data),
                "transcription": await self._transcribe_audio(input_data.data),
                "speakers": self._identify_speakers(input_data.data),
                "emotions": self._detect_emotions(input_data.data),
                "quality": self._analyze_audio_quality(input_data.data),
            }

            processing_time = (datetime.now() - start_time).total_seconds()

            return PerceptionResult(
                input_id=f"{input_data.modality.value}_{hash(str(input_data.data)) % 10000}",
                modality=input_data.modality,
                status=ProcessingStatus.COMPLETED,
                extracted_data=results,
                confidence_scores={
                    "overall": 0.7,
                    "transcription": 0.75,
                    "speakers": 0.6,
                    "emotions": 0.65,
                },
                processing_time=processing_time,
                metadata=input_data.metadata,
            )

        except Exception as exc:
            processing_time = (datetime.now() - start_time).total_seconds()
            return PerceptionResult(
                input_id=f"{input_data.modality.value}_error",
                modality=input_data.modality,
                status=ProcessingStatus.FAILED,
                error_message=str(exc),
                processing_time=processing_time,
            )

    def _get_audio_duration(self, audio_data: Any) -> float:
        """Get audio duration in seconds."""
        # Placeholder - in real implementation would analyze audio file
        return 0.0

    async def _transcribe_audio(self, audio_data: Any) -> str:
        """Transcribe speech to text."""
        if not self.is_available or self.speech_recognition is None:
            return ""

        try:
            # Placeholder transcription
            return "Audio transcription not implemented"
        except Exception:
            return ""

    def _identify_speakers(self, audio_data: Any) -> List[Dict[str, Any]]:
        """Identify speakers in audio."""
        return [{"speaker_id": "unknown", "segments": []}]

    def _detect_emotions(self, audio_data: Any) -> Dict[str, float]:
        """Detect emotions in audio."""
        return {"neutral": 0.6, "happy": 0.2, "sad": 0.1, "angry": 0.1}

    def _analyze_audio_quality(self, audio_data: Any) -> Dict[str, Any]:
        """Analyze audio quality metrics."""
        return {
            "snr": 20.0,  # Signal-to-noise ratio
            "quality": "good",
            "artifacts": [],
        }


class DocumentProcessor(PerceptionProcessor):
    """Document processing and intelligence."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(ModalityType.DOCUMENT, config)
        self.is_available = True
        self.capabilities = [
            "text_extraction",
            "layout_analysis",
            "table_extraction",
            "form_recognition",
            "document_classification",
            "key_value_extraction",
        ]

    async def process(self, input_data: PerceptionInput) -> PerceptionResult:
        """Process document input."""
        start_time = datetime.now()

        try:
            # Extract document content
            content = self._extract_document_content(input_data.data, input_data.format)

            results = {
                "text": content.get("text", ""),
                "tables": content.get("tables", []),
                "forms": content.get("forms", []),
                "layout": content.get("layout", {}),
                "metadata": content.get("metadata", {}),
                "classification": self._classify_document(content),
            }

            processing_time = (datetime.now() - start_time).total_seconds()

            return PerceptionResult(
                input_id=f"{input_data.modality.value}_{hash(str(input_data.data)) % 10000}",
                modality=input_data.modality,
                status=ProcessingStatus.COMPLETED,
                extracted_data=results,
                confidence_scores={
                    "overall": 0.8,
                    "text_extraction": 0.85,
                    "layout": 0.75,
                    "classification": 0.7,
                },
                processing_time=processing_time,
                metadata=input_data.metadata,
            )

        except Exception as exc:
            processing_time = (datetime.now() - start_time).total_seconds()
            return PerceptionResult(
                input_id=f"{input_data.modality.value}_error",
                modality=input_data.modality,
                status=ProcessingStatus.FAILED,
                error_message=str(exc),
                processing_time=processing_time,
            )

    def _extract_document_content(
        self, data: Any, format: Optional[str]
    ) -> Dict[str, Any]:
        """Extract content from document."""
        if isinstance(data, str):
            return {
                "text": data,
                "tables": [],
                "forms": [],
                "layout": {"type": "plain_text"},
                "metadata": {"format": "text"},
            }
        elif isinstance(data, bytes):
            # Try to extract text from binary data
            try:
                text = data.decode("utf-8", errors="ignore")
                return {
                    "text": text,
                    "tables": [],
                    "forms": [],
                    "layout": {"type": "binary_text"},
                    "metadata": {"format": format or "unknown"},
                }
            except Exception:
                return {
                    "text": "",
                    "tables": [],
                    "forms": [],
                    "layout": {"type": "binary"},
                    "metadata": {
                        "format": format or "unknown",
                        "error": "could_not_decode",
                    },
                }
        else:
            return {
                "text": str(data),
                "tables": [],
                "forms": [],
                "layout": {"type": "unknown"},
                "metadata": {"format": "unknown"},
            }

    def _classify_document(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """Classify document type."""
        text = content.get("text", "").lower()

        # Simple keyword-based classification
        if "invoice" in text or "bill" in text:
            doc_type = "invoice"
        elif "contract" in text or "agreement" in text:
            doc_type = "contract"
        elif "resume" in text or "cv" in text:
            doc_type = "resume"
        elif "report" in text:
            doc_type = "report"
        else:
            doc_type = "document"

        return {"type": doc_type, "confidence": 0.6, "keywords": []}


class MultimodalPerceptionSuite:
    """
    Multimodal perception suite.
    """

    def __init__(
        self,
        config_manager=None,
        *,
        enable_parallel_processing: bool = True,
        max_concurrent_tasks: int = 5,
        processing_timeout: int = 30,
    ):
        self.config_manager = config_manager
        self.enable_parallel_processing = enable_parallel_processing
        self.max_concurrent_tasks = max_concurrent_tasks
        self.processing_timeout = processing_timeout

        # Initialize processors
        self.processors: Dict[ModalityType, PerceptionProcessor] = {}
        self._initialize_processors()

        # Processing management
        self.processing_lock = threading.Lock()
        self.active_tasks: Dict[str, asyncio.Task] = {}
        self.task_semaphore = asyncio.Semaphore(max_concurrent_tasks)

        # Results storage
        self.results_cache: Dict[str, PerceptionResult] = {}
        self.fusion_cache: Dict[str, MultimodalContext] = {}

        logger.info(
            "Multimodal Perception Suite initialized with %d processors",
            len(self.processors),
        )

    def _initialize_processors(self):
        """Initialize all perception processors."""
        processor_classes = {
            ModalityType.TEXT: TextProcessor,
            ModalityType.IMAGE: ImageProcessor,
            ModalityType.AUDIO: AudioProcessor,
            ModalityType.DOCUMENT: DocumentProcessor,
        }

        for modality, processor_class in processor_classes.items():
            try:
                processor = processor_class()
                if processor.check_availability():
                    self.processors[modality] = processor
                    logger.info("Initialized %s processor", modality.value)
                else:
                    logger.warning("%s processor not available", modality.value)
            except Exception as exc:
                logger.error(
                    "Failed to initialize %s processor: %s", modality.value, exc
                )

    async def process_input(
        self, input_data: PerceptionInput, enable_fusion: bool = True
    ) -> Union[PerceptionResult, MultimodalContext]:
        """
        Process a single perception input.

        Args:
            input_data: Input data to process
            enable_fusion: Whether to enable multimodal fusion

        Returns:
            Perception result or multimodal context
        """
        async with self.task_semaphore:
            try:
                # Check cache first
                cache_key = (
                    f"{input_data.modality.value}_{hash(str(input_data.data)) % 10000}"
                )
                if cache_key in self.results_cache:
                    cached_result = self.results_cache[cache_key]
                    if enable_fusion:
                        return await self._fuse_with_context(cached_result)
                    return cached_result

                # Get appropriate processor
                processor = self.processors.get(input_data.modality)
                if not processor:
                    return PerceptionResult(
                        input_id=f"{input_data.modality.value}_no_processor",
                        modality=input_data.modality,
                        status=ProcessingStatus.FAILED,
                        error_message=f"No processor available for {input_data.modality.value}",
                    )

                # Process input
                result = await asyncio.wait_for(
                    processor.process(input_data), timeout=self.processing_timeout
                )

                # Cache result
                self.results_cache[cache_key] = result

                # Apply fusion if enabled
                if enable_fusion:
                    return await self._fuse_with_context(result)

                return result

            except asyncio.TimeoutError:
                return PerceptionResult(
                    input_id=f"{input_data.modality.value}_timeout",
                    modality=input_data.modality,
                    status=ProcessingStatus.FAILED,
                    error_message=f"Processing timeout after {self.processing_timeout}s",
                )
            except Exception as exc:
                logger.error("Processing error: %s", exc)
                return PerceptionResult(
                    input_id=f"{input_data.modality.value}_error",
                    modality=input_data.modality,
                    status=ProcessingStatus.FAILED,
                    error_message=str(exc),
                )

    async def process_multimodal(
        self, inputs: List[PerceptionInput], fusion_strategy: str = "complementary"
    ) -> MultimodalContext:
        """
        Process multiple inputs and fuse results.

        Args:
            inputs: List of perception inputs
            fusion_strategy: Strategy for fusing results

        Returns:
            Fused multimodal context
        """
        if not inputs:
            return MultimodalContext()

        # Process all inputs (in parallel if enabled)
        if self.enable_parallel_processing:
            tasks = [self.process_input(inp, enable_fusion=False) for inp in inputs]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            # Filter out exceptions and keep only successful results
            valid_results = [
                r
                for r in results
                if isinstance(r, PerceptionResult)
                and r.status == ProcessingStatus.COMPLETED
            ]
        else:
            valid_results = []
            for inp in inputs:
                result = await self.process_input(inp, enable_fusion=False)
                if (
                    isinstance(result, PerceptionResult)
                    and result.status == ProcessingStatus.COMPLETED
                ):
                    valid_results.append(result)

        # Create multimodal context
        context = MultimodalContext(inputs=valid_results)

        # Apply fusion strategy
        if fusion_strategy == "complementary":
            context = await self._apply_complementary_fusion(context)
        elif fusion_strategy == "redundant":
            context = await self._apply_redundant_fusion(context)
        elif fusion_strategy == "collaborative":
            context = await self._apply_collaborative_fusion(context)

        return context

    async def _fuse_with_context(self, result: PerceptionResult) -> MultimodalContext:
        """Fuse a single result with available context."""
        # Simple context fusion - in real implementation would be more sophisticated
        context = MultimodalContext(inputs=[result])

        # Add basic cross-modal relations
        relations = []
        if result.modality == ModalityType.TEXT:
            # Look for related image/audio results in recent cache
            for cache_result in list(self.results_cache.values())[
                -10:
            ]:  # Last 10 results
                if cache_result.modality in [ModalityType.IMAGE, ModalityType.AUDIO]:
                    relations.append(
                        {
                            "source": result.input_id,
                            "target": cache_result.input_id,
                            "relation": "temporal_association",
                            "strength": 0.5,
                        }
                    )

        context.cross_modal_relations = relations
        context.confidence = result.confidence_scores.get("overall", 0.5)

        return context

    async def _apply_complementary_fusion(
        self, context: MultimodalContext
    ) -> MultimodalContext:
        """Apply complementary fusion strategy."""
        # Combine complementary information from different modalities
        fused_insights = {}

        text_results = [r for r in context.inputs if r.modality == ModalityType.TEXT]
        image_results = [r for r in context.inputs if r.modality == ModalityType.IMAGE]
        audio_results = [r for r in context.inputs if r.modality == ModalityType.AUDIO]

        # Example: Combine text sentiment with audio emotion
        if text_results and audio_results:
            text_sentiment = text_results[0].extracted_data.get("sentiment", {})
            audio_emotion = audio_results[0].extracted_data.get("emotions", {})

            # Fuse sentiment and emotion
            fused_sentiment = self._fuse_sentiment_and_emotion(
                text_sentiment, audio_emotion
            )
            fused_insights["fused_sentiment"] = fused_sentiment

        # Example: Combine image description with text content
        if image_results and text_results:
            image_desc = image_results[0].extracted_data.get("description", "")
            text_content = text_results[0].extracted_data.get("text", "")

            fused_insights["combined_description"] = f"{image_desc}. {text_content}"

        context.fused_insights = fused_insights
        context.confidence = sum(
            r.confidence_scores.get("overall", 0) for r in context.inputs
        ) / len(context.inputs)

        return context

    async def _apply_redundant_fusion(
        self, context: MultimodalContext
    ) -> MultimodalContext:
        """Apply redundant fusion strategy (combine similar information)."""
        # Combine redundant information for higher confidence
        context.fused_insights = {"fusion_type": "redundant"}
        context.confidence = min(
            1.0,
            sum(r.confidence_scores.get("overall", 0) for r in context.inputs)
            / len(context.inputs)
            * 1.2,
        )
        return context

    async def _apply_collaborative_fusion(
        self, context: MultimodalContext
    ) -> MultimodalContext:
        """Apply collaborative fusion strategy."""
        # Advanced fusion using all modalities together
        context.fused_insights = {"fusion_type": "collaborative"}
        context.confidence = sum(
            r.confidence_scores.get("overall", 0) for r in context.inputs
        ) / len(context.inputs)
        return context

    def _fuse_sentiment_and_emotion(
        self, text_sentiment: Dict, audio_emotion: Dict
    ) -> Dict:
        """Fuse text sentiment with audio emotion."""
        # Simple fusion logic
        text_score = text_sentiment.get("score", 0.5)
        audio_happy = audio_emotion.get("happy", 0)
        audio_sad = audio_emotion.get("sad", 0)

        # Combine scores
        combined_score = (text_score + (audio_happy - audio_sad + 1) / 2) / 2

        if combined_score > 0.6:
            sentiment = "positive"
        elif combined_score < 0.4:
            sentiment = "negative"
        else:
            sentiment = "neutral"

        return {
            "sentiment": sentiment,
            "score": combined_score,
            "sources": ["text", "audio"],
        }

    def get_available_modalities(self) -> List[ModalityType]:
        """Get list of available modalities."""
        return list(self.processors.keys())

    def get_processor_capabilities(self, modality: ModalityType) -> List[str]:
        """Get capabilities for a specific modality."""
        processor = self.processors.get(modality)
        return processor.get_capabilities() if processor else []

    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        return {
            "available_modalities": len(self.processors),
            "cached_results": len(self.results_cache),
            "active_tasks": len(self.active_tasks),
            "processors": {
                modality.value: {
                    "available": processor.check_availability(),
                    "capabilities": processor.get_capabilities(),
                }
                for modality, processor in self.processors.items()
            },
        }

    def clear_cache(self):
        """Clear all caches."""
        self.results_cache.clear()
        self.fusion_cache.clear()
        logger.info("Cleared perception caches")


# Register with provider registry
def create_multimodal_perception_suite(config_manager=None, **kwargs):
    """Factory function for MultimodalPerceptionSuite."""
    return MultimodalPerceptionSuite(config_manager=config_manager, **kwargs)


provider_registry.register_lazy(
    "perception",
    "multimodal_suite",
    "mia.adaptive_intelligence.multimodal_perception_suite",
    "create_multimodal_perception_suite",
    default=True,
)
