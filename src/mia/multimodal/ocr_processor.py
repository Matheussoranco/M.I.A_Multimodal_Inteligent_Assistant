"""
OCR Processor: Optical Character Recognition with LLM integration.
Supports multiple OCR engines: Tesseract, EasyOCR, PaddleOCR, TrOCR, and cloud APIs.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image

from ..config_manager import ConfigManager
from ..error_handler import global_error_handler, with_error_handling
from ..exceptions import (
    ConfigurationError,
    InitializationError,
)

# Optional imports with error handling
try:
    import cv2

    HAS_OPENCV = True
except ImportError:
    cv2 = None
    HAS_OPENCV = False

try:
    import pytesseract

    HAS_TESSERACT = True
except ImportError:
    pytesseract = None
    HAS_TESSERACT = False

try:
    import easyocr

    HAS_EASYOCR = True
except ImportError:
    easyocr = None
    HAS_EASYOCR = False

try:
    from paddleocr import PaddleOCR

    HAS_PADDLEOCR = True
except ImportError:
    PaddleOCR = None
    HAS_PADDLEOCR = False

try:
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    import torch

    HAS_TROCR = True
except ImportError:
    TrOCRProcessor = None
    VisionEncoderDecoderModel = None
    torch = None
    HAS_TROCR = False

try:
    import requests

    HAS_REQUESTS = True
except ImportError:
    requests = None
    HAS_REQUESTS = False

logger = logging.getLogger(__name__)


SUPPORTED_OCR_PROVIDERS = {
    "tesseract",
    "easyocr",
    "paddleocr",
    "trocr",
    "openai",  # GPT-4 Vision
    "anthropic",  # Claude Vision
    "google",  # Google Cloud Vision
    "azure",  # Azure Computer Vision
    "auto",
}

# Default models per provider
DEFAULT_OCR_MODELS = {
    "trocr": "microsoft/trocr-base-printed",
    "openai": "gpt-4o",
    "anthropic": "claude-3-haiku-20240307",
}


@dataclass
class OCRConfig:
    """Configuration for OCR processing."""
    
    provider: str = "auto"
    languages: List[str] = field(default_factory=lambda: ["en"])
    model_id: Optional[str] = None
    api_key: Optional[str] = None
    device: str = "auto"
    confidence_threshold: float = 0.5
    preprocessing: bool = True
    deskew: bool = True
    denoise: bool = True
    binarize: bool = False
    enhance_contrast: bool = True
    detect_orientation: bool = True
    segment_mode: str = "auto"  # auto, single_block, single_line, sparse
    output_format: str = "text"  # text, structured, hocr
    
    def validate(self) -> None:
        """Validate OCR configuration."""
        if self.provider not in SUPPORTED_OCR_PROVIDERS:
            raise ConfigurationError(
                f"Unsupported OCR provider: {self.provider}",
                "UNSUPPORTED_OCR_PROVIDER",
            )
        
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ConfigurationError(
                "Confidence threshold must be between 0.0 and 1.0",
                "INVALID_CONFIDENCE_THRESHOLD",
            )


@dataclass
class OCRResult:
    """Result from OCR processing."""
    
    text: str
    confidence: float
    boxes: List[Dict[str, Any]] = field(default_factory=list)
    language: Optional[str] = None
    orientation: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "text": self.text,
            "confidence": self.confidence,
            "boxes": self.boxes,
            "language": self.language,
            "orientation": self.orientation,
            "metadata": self.metadata,
        }


class OCRProcessor:
    """Unified OCR processor supporting multiple engines with LLM integration."""

    @classmethod
    def detect_available_providers(cls) -> List[Dict[str, Any]]:
        """Detect available OCR providers."""
        providers = []
        
        if HAS_TESSERACT:
            try:
                # Check if tesseract is actually installed
                pytesseract.get_tesseract_version()
                providers.append({
                    "name": "tesseract",
                    "type": "local",
                    "available": True,
                    "languages": cls._get_tesseract_languages(),
                })
            except Exception:
                pass
        
        if HAS_EASYOCR:
            providers.append({
                "name": "easyocr",
                "type": "local",
                "available": True,
            })
        
        if HAS_PADDLEOCR:
            providers.append({
                "name": "paddleocr",
                "type": "local",
                "available": True,
            })
        
        if HAS_TROCR:
            providers.append({
                "name": "trocr",
                "type": "local",
                "available": True,
                "model": DEFAULT_OCR_MODELS["trocr"],
            })
        
        # Check cloud providers
        if os.getenv("OPENAI_API_KEY"):
            providers.append({
                "name": "openai",
                "type": "api",
                "available": True,
                "model": DEFAULT_OCR_MODELS["openai"],
            })
        
        if os.getenv("ANTHROPIC_API_KEY"):
            providers.append({
                "name": "anthropic",
                "type": "api",
                "available": True,
                "model": DEFAULT_OCR_MODELS["anthropic"],
            })
        
        if os.getenv("GOOGLE_APPLICATION_CREDENTIALS") or os.getenv("GOOGLE_API_KEY"):
            providers.append({
                "name": "google",
                "type": "api",
                "available": True,
            })
        
        if os.getenv("AZURE_COMPUTER_VISION_KEY"):
            providers.append({
                "name": "azure",
                "type": "api",
                "available": True,
            })
        
        return providers

    @classmethod
    def _get_tesseract_languages(cls) -> List[str]:
        """Get available Tesseract languages."""
        try:
            return pytesseract.get_languages()
        except Exception:
            return ["eng"]

    @classmethod
    def interactive_provider_selection(cls) -> Dict[str, Any]:
        """Interactive OCR provider selection."""
        print("\n" + "═" * 60)
        print("📝 M.I.A - OCR Provider Selection")
        print("═" * 60)
        
        providers = cls.detect_available_providers()
        
        if not providers:
            print("\n⚠️ No OCR providers available!")
            print("💡 Install options:")
            print("   - pip install pytesseract (+ install Tesseract)")
            print("   - pip install easyocr")
            print("   - pip install paddleocr")
            raise ConfigurationError(
                "No OCR providers available",
                "NO_OCR_PROVIDERS",
            )
        
        print("\n📋 Available OCR Providers:")
        for i, provider in enumerate(providers, 1):
            type_icon = "🖥️" if provider["type"] == "local" else "🌐"
            print(f"  {i}. {type_icon} {provider['name'].upper()}")
        
        while True:
            try:
                choice = input(f"\nSelect provider [1-{len(providers)}] (default: 1): ").strip() or "1"
                idx = int(choice) - 1
                if 0 <= idx < len(providers):
                    selected = providers[idx]
                    break
                print("❌ Invalid choice.")
            except (ValueError, KeyboardInterrupt, EOFError):
                idx = 0
                selected = providers[0]
                break
        
        print(f"\n✅ Selected: {selected['name'].upper()}")
        
        return selected

    def __init__(
        self,
        provider: Optional[str] = None,
        languages: Optional[List[str]] = None,
        model_id: Optional[str] = None,
        api_key: Optional[str] = None,
        config_manager: Optional[ConfigManager] = None,
        auto_detect: bool = True,
        device: str = "auto",
        preprocessing: bool = True,
        llm_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the OCR processor.
        
        Args:
            provider: OCR provider (tesseract, easyocr, etc.)
            languages: List of language codes
            model_id: Model identifier for ML-based OCR
            api_key: API key for cloud providers
            config_manager: Optional ConfigManager instance
            auto_detect: Whether to auto-detect providers
            device: Device to use (auto, cpu, cuda)
            preprocessing: Whether to preprocess images
            llm_manager: Optional LLM manager for enhanced OCR
        """
        self.config_manager = config_manager or ConfigManager()
        self.llm_manager = llm_manager
        
        # Auto-detect if needed
        import sys
        is_testing = "pytest" in sys.modules or os.getenv("TESTING") == "true"
        
        if auto_detect and not provider and not is_testing:
            try:
                detected = self.interactive_provider_selection()
                provider = detected["name"]
                logger.info(f"Auto-detected OCR provider: {provider}")
            except Exception as e:
                logger.warning(f"Auto-detection failed: {e}")
                # Default to first available
                providers = self.detect_available_providers()
                if providers:
                    provider = providers[0]["name"]
        
        self.provider = provider or "tesseract"
        self.languages = languages or ["en"]
        self.model_id = model_id or DEFAULT_OCR_MODELS.get(self.provider)
        self.api_key = api_key
        self.device = self._resolve_device(device)
        self.preprocessing = preprocessing
        
        # Initialize components
        self._reader: Optional[Any] = None
        self._model: Optional[Any] = None
        self._processor: Optional[Any] = None
        self._available = True
        
        try:
            self._initialize_provider()
        except Exception as e:
            logger.warning(f"Failed to initialize OCR provider {self.provider}: {e}")
            self._available = False
            if is_testing:
                raise

    def _resolve_device(self, device: str) -> str:
        """Resolve the device to use."""
        if device != "auto":
            return device
        
        if HAS_TROCR and torch is not None:
            if torch.cuda.is_available():
                return "cuda"
        return "cpu"

    def _initialize_provider(self) -> None:
        """Initialize the specific provider."""
        if self.provider == "tesseract":
            self._initialize_tesseract()
        elif self.provider == "easyocr":
            self._initialize_easyocr()
        elif self.provider == "paddleocr":
            self._initialize_paddleocr()
        elif self.provider == "trocr":
            self._initialize_trocr()
        elif self.provider in ("openai", "anthropic", "google", "azure"):
            self._initialize_cloud_provider()
        else:
            raise ConfigurationError(
                f"Unknown OCR provider: {self.provider}",
                "UNKNOWN_PROVIDER",
            )

    def _initialize_tesseract(self) -> None:
        """Initialize Tesseract OCR."""
        if not HAS_TESSERACT:
            raise InitializationError(
                "pytesseract not installed. Run: pip install pytesseract",
                "MISSING_DEPENDENCY",
            )
        
        try:
            pytesseract.get_tesseract_version()  # type: ignore[union-attr]
        except Exception as e:
            raise InitializationError(
                f"Tesseract not installed or not in PATH: {e}",
                "TESSERACT_NOT_FOUND",
            )

    def _initialize_easyocr(self) -> None:
        """Initialize EasyOCR."""
        if not HAS_EASYOCR:
            raise InitializationError(
                "easyocr not installed. Run: pip install easyocr",
                "MISSING_DEPENDENCY",
            )
        
        gpu = self.device == "cuda"
        self._reader = easyocr.Reader(self.languages, gpu=gpu)

    def _initialize_paddleocr(self) -> None:
        """Initialize PaddleOCR."""
        if not HAS_PADDLEOCR:
            raise InitializationError(
                "paddleocr not installed. Run: pip install paddleocr",
                "MISSING_DEPENDENCY",
            )
        
        use_gpu = self.device == "cuda"
        lang = self.languages[0] if self.languages else "en"
        self._reader = PaddleOCR(use_angle_cls=True, lang=lang, use_gpu=use_gpu)

    def _initialize_trocr(self) -> None:
        """Initialize TrOCR transformer model."""
        if not HAS_TROCR:
            raise InitializationError(
                "transformers not installed. Run: pip install transformers torch",
                "MISSING_DEPENDENCY",
            )
        
        model_id = self.model_id or DEFAULT_OCR_MODELS["trocr"]
        self._processor = TrOCRProcessor.from_pretrained(model_id)
        self._model = VisionEncoderDecoderModel.from_pretrained(model_id)
        
        if self.device == "cuda" and torch.cuda.is_available():
            self._model = self._model.cuda()
        
        self._model.eval()

    def _initialize_cloud_provider(self) -> None:
        """Initialize cloud OCR provider."""
        if self.provider == "openai":
            self.api_key = self.api_key or os.getenv("OPENAI_API_KEY")
        elif self.provider == "anthropic":
            self.api_key = self.api_key or os.getenv("ANTHROPIC_API_KEY")
        elif self.provider == "google":
            self.api_key = self.api_key or os.getenv("GOOGLE_API_KEY")
        elif self.provider == "azure":
            self.api_key = self.api_key or os.getenv("AZURE_COMPUTER_VISION_KEY")
        
        if not self.api_key and self.provider not in ("google",):
            raise ConfigurationError(
                f"API key required for {self.provider}",
                "MISSING_API_KEY",
            )

    @property
    def is_available(self) -> bool:
        """Check if the OCR processor is available."""
        return self._available

    def preprocess_image(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        deskew: bool = True,
        denoise: bool = True,
        binarize: bool = False,
        enhance_contrast: bool = True,
    ) -> np.ndarray:
        """Preprocess image for better OCR results.
        
        Args:
            image: Input image (path, PIL Image, or numpy array)
            deskew: Whether to correct image skew
            denoise: Whether to remove noise
            binarize: Whether to convert to binary
            enhance_contrast: Whether to enhance contrast
            
        Returns:
            Preprocessed image as numpy array
        """
        # Load image
        if isinstance(image, (str, Path)):
            img = Image.open(image)
            img_array = np.array(img)
        elif isinstance(image, Image.Image):
            img_array = np.array(image)
        else:
            img_array = image
        
        # Convert to grayscale if color
        if len(img_array.shape) == 3:
            if HAS_OPENCV:
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                # Simple grayscale conversion without OpenCV
                gray = np.mean(img_array, axis=2).astype(np.uint8)
        else:
            gray = img_array
        
        if not HAS_OPENCV:
            # Return grayscale if OpenCV not available
            return gray
        
        # Denoise
        if denoise:
            gray = cv2.fastNlMeansDenoising(gray)
        
        # Enhance contrast
        if enhance_contrast:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)
        
        # Deskew
        if deskew:
            gray = self._deskew_image(gray)
        
        # Binarize
        if binarize:
            gray = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )
        
        return gray

    def _deskew_image(self, image: np.ndarray) -> np.ndarray:
        """Deskew an image."""
        if not HAS_OPENCV:
            return image
        
        try:
            # Find edges
            edges = cv2.Canny(image, 50, 150, apertureSize=3)
            
            # Find lines using Hough transform
            lines = cv2.HoughLinesP(
                edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10
            )
            
            if lines is None or len(lines) == 0:
                return image
            
            # Calculate average angle
            angles = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
                if -45 < angle < 45:  # Filter horizontal-ish lines
                    angles.append(angle)
            
            if not angles:
                return image
            
            median_angle = float(np.median(angles))
            
            # Rotate image
            (h, w) = image.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
            rotated = cv2.warpAffine(
                image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
            )
            
            return rotated
        except Exception as e:
            logger.warning(f"Deskew failed: {e}")
            return image

    def extract_text(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        preprocess: Optional[bool] = None,
        language: Optional[str] = None,
        confidence_threshold: float = 0.5,
        return_boxes: bool = False,
    ) -> Union[str, OCRResult]:
        """Extract text from an image.
        
        Args:
            image: Input image (path, PIL Image, or numpy array)
            preprocess: Whether to preprocess (overrides init setting)
            language: Override language
            confidence_threshold: Minimum confidence for text
            return_boxes: Whether to return bounding boxes
            
        Returns:
            Extracted text or OCRResult with boxes
        """
        do_preprocess = preprocess if preprocess is not None else self.preprocessing
        
        # Load and optionally preprocess
        if isinstance(image, (str, Path)):
            img = Image.open(image)
        elif isinstance(image, np.ndarray):
            img = Image.fromarray(image)
        else:
            img = image
        
        if do_preprocess:
            img_array = self.preprocess_image(img)
            img = Image.fromarray(img_array) if len(img_array.shape) == 2 else Image.fromarray(img_array)
        
        # Route to appropriate provider
        if self.provider == "tesseract":
            result = self._ocr_tesseract(img, language, return_boxes)
        elif self.provider == "easyocr":
            result = self._ocr_easyocr(img, return_boxes)
        elif self.provider == "paddleocr":
            result = self._ocr_paddleocr(img, return_boxes)
        elif self.provider == "trocr":
            result = self._ocr_trocr(img)
        elif self.provider in ("openai", "anthropic"):
            result = self._ocr_llm_vision(img)
        elif self.provider == "google":
            result = self._ocr_google_vision(img)
        elif self.provider == "azure":
            result = self._ocr_azure_vision(img)
        else:
            raise ConfigurationError(
                f"OCR not implemented for provider: {self.provider}",
                "NOT_IMPLEMENTED",
            )
        
        # Filter by confidence
        if isinstance(result, OCRResult) and result.boxes:
            result.boxes = [
                box for box in result.boxes
                if box.get("confidence", 1.0) >= confidence_threshold
            ]
            result.text = " ".join([box.get("text", "") for box in result.boxes])
        
        if return_boxes:
            return result
        return result.text if isinstance(result, OCRResult) else result

    def _ocr_tesseract(
        self, img: Image.Image, language: Optional[str], return_boxes: bool
    ) -> OCRResult:
        """Perform OCR using Tesseract."""
        lang = language or "+".join(self.languages)
        
        if return_boxes:
            # Get detailed output with boxes
            data = pytesseract.image_to_data(img, lang=lang, output_type=pytesseract.Output.DICT)
            
            boxes = []
            full_text = []
            
            for i in range(len(data["text"])):
                text = data["text"][i].strip()
                conf = float(data["conf"][i])
                
                if text and conf > 0:
                    boxes.append({
                        "text": text,
                        "confidence": conf / 100.0,
                        "bbox": {
                            "x": data["left"][i],
                            "y": data["top"][i],
                            "width": data["width"][i],
                            "height": data["height"][i],
                        },
                        "level": data["level"][i],
                        "block_num": data["block_num"][i],
                        "line_num": data["line_num"][i],
                        "word_num": data["word_num"][i],
                    })
                    full_text.append(text)
            
            avg_conf = float(np.mean([b["confidence"] for b in boxes])) if boxes else 0.0
            
            return OCRResult(
                text=" ".join(full_text),
                confidence=avg_conf,
                boxes=boxes,
                language=lang,
            )
        else:
            text = pytesseract.image_to_string(img, lang=lang)
            return OCRResult(text=text.strip(), confidence=1.0, language=lang)

    def _ocr_easyocr(self, img: Image.Image, return_boxes: bool) -> OCRResult:
        """Perform OCR using EasyOCR."""
        img_array = np.array(img)
        results = self._reader.readtext(img_array)
        
        boxes = []
        full_text = []
        
        for bbox, text, conf in results:
            # Convert bbox to standard format
            x_coords = [p[0] for p in bbox]
            y_coords = [p[1] for p in bbox]
            
            boxes.append({
                "text": text,
                "confidence": conf,
                "bbox": {
                    "x": min(x_coords),
                    "y": min(y_coords),
                    "width": max(x_coords) - min(x_coords),
                    "height": max(y_coords) - min(y_coords),
                },
                "polygon": bbox,
            })
            full_text.append(text)
        
        avg_conf = float(np.mean([b["confidence"] for b in boxes])) if boxes else 0.0
        
        return OCRResult(
            text=" ".join(full_text),
            confidence=avg_conf,
            boxes=boxes,
        )

    def _ocr_paddleocr(self, img: Image.Image, return_boxes: bool) -> OCRResult:
        """Perform OCR using PaddleOCR."""
        img_array = np.array(img)
        results = self._reader.ocr(img_array, cls=True)
        
        boxes = []
        full_text = []
        
        if results and results[0]:
            for line in results[0]:
                bbox, (text, conf) = line
                
                x_coords = [p[0] for p in bbox]
                y_coords = [p[1] for p in bbox]
                
                boxes.append({
                    "text": text,
                    "confidence": conf,
                    "bbox": {
                        "x": min(x_coords),
                        "y": min(y_coords),
                        "width": max(x_coords) - min(x_coords),
                        "height": max(y_coords) - min(y_coords),
                    },
                    "polygon": bbox,
                })
                full_text.append(text)
        
        avg_conf = float(np.mean([b["confidence"] for b in boxes])) if boxes else 0.0
        
        return OCRResult(
            text=" ".join(full_text),
            confidence=avg_conf,
            boxes=boxes,
        )

    def _ocr_trocr(self, img: Image.Image) -> OCRResult:
        """Perform OCR using TrOCR transformer model."""
        # Convert to RGB if necessary
        if img.mode != "RGB":
            img = img.convert("RGB")
        
        # Process image
        pixel_values = self._processor(images=img, return_tensors="pt").pixel_values
        
        if self.device == "cuda":
            pixel_values = pixel_values.cuda()
        
        # Generate text
        with torch.no_grad():
            generated_ids = self._model.generate(pixel_values)
        
        text = self._processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        return OCRResult(
            text=text or "",
            confidence=1.0,  # TrOCR doesn't provide confidence
            metadata={"model": self.model_id},
        )

    def _ocr_llm_vision(self, img: Image.Image) -> OCRResult:
        """Perform OCR using LLM vision capabilities (OpenAI/Anthropic)."""
        # Convert image to base64
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()
        
        if self.provider == "openai":
            return self._ocr_openai_vision(img_base64)
        elif self.provider == "anthropic":
            return self._ocr_anthropic_vision(img_base64)
        else:
            raise ConfigurationError(f"LLM vision not supported for {self.provider}")

    def _ocr_openai_vision(self, img_base64: str) -> OCRResult:
        """Perform OCR using OpenAI Vision."""
        from openai import OpenAI
        
        client = OpenAI(api_key=self.api_key)
        
        response = client.chat.completions.create(
            model=self.model_id or "gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Extract all text from this image. Return only the extracted text, preserving the original formatting and structure as much as possible.",
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{img_base64}",
                            },
                        },
                    ],
                }
            ],
            max_tokens=4096,
        )
        
        text = response.choices[0].message.content
        
        return OCRResult(
            text=text or "",
            confidence=1.0,
            metadata={"model": self.model_id, "provider": "openai"},
        )

    def _ocr_anthropic_vision(self, img_base64: str) -> OCRResult:
        """Perform OCR using Anthropic Claude Vision."""
        import requests as req
        
        response = req.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model_id or "claude-3-haiku-20240307",
                "max_tokens": 4096,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": img_base64,
                                },
                            },
                            {
                                "type": "text",
                                "text": "Extract all text from this image. Return only the extracted text, preserving the original formatting and structure as much as possible.",
                            },
                        ],
                    }
                ],
            },
            timeout=60,
        )
        response.raise_for_status()
        data = response.json()
        
        text = data["content"][0]["text"]
        
        return OCRResult(
            text=text,
            confidence=1.0,
            metadata={"model": self.model_id, "provider": "anthropic"},
        )

    def _ocr_google_vision(self, img: Image.Image) -> OCRResult:
        """Perform OCR using Google Cloud Vision."""
        # Convert image to base64
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()
        
        api_key = self.api_key or os.getenv("GOOGLE_API_KEY")
        
        response = requests.post(
            f"https://vision.googleapis.com/v1/images:annotate?key={api_key}",
            json={
                "requests": [
                    {
                        "image": {"content": img_base64},
                        "features": [{"type": "TEXT_DETECTION"}],
                    }
                ]
            },
            timeout=60,
        )
        response.raise_for_status()
        data = response.json()
        
        if "responses" in data and data["responses"]:
            annotations = data["responses"][0].get("textAnnotations", [])
            if annotations:
                full_text = annotations[0].get("description", "")
                
                boxes = []
                for ann in annotations[1:]:  # Skip first (full text)
                    vertices = ann.get("boundingPoly", {}).get("vertices", [])
                    if vertices:
                        boxes.append({
                            "text": ann.get("description", ""),
                            "confidence": 1.0,
                            "bbox": {
                                "x": vertices[0].get("x", 0),
                                "y": vertices[0].get("y", 0),
                                "width": vertices[2].get("x", 0) - vertices[0].get("x", 0),
                                "height": vertices[2].get("y", 0) - vertices[0].get("y", 0),
                            },
                        })
                
                return OCRResult(
                    text=full_text,
                    confidence=1.0,
                    boxes=boxes,
                    metadata={"provider": "google"},
                )
        
        return OCRResult(text="", confidence=0.0)

    def _ocr_azure_vision(self, img: Image.Image) -> OCRResult:
        """Perform OCR using Azure Computer Vision."""
        # Convert image to bytes
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        img_bytes = buffered.getvalue()
        
        endpoint = os.getenv("AZURE_COMPUTER_VISION_ENDPOINT", "").rstrip("/")
        
        response = requests.post(
            f"{endpoint}/vision/v3.2/ocr",
            headers={
                "Ocp-Apim-Subscription-Key": self.api_key,
                "Content-Type": "application/octet-stream",
            },
            params={"language": self.languages[0] if self.languages else "en"},
            data=img_bytes,
            timeout=60,
        )
        response.raise_for_status()
        data = response.json()
        
        full_text = []
        boxes = []
        
        for region in data.get("regions", []):
            for line in region.get("lines", []):
                line_text = " ".join([word["text"] for word in line.get("words", [])])
                full_text.append(line_text)
                
                for word in line.get("words", []):
                    bbox_parts = word.get("boundingBox", "0,0,0,0").split(",")
                    boxes.append({
                        "text": word["text"],
                        "confidence": 1.0,
                        "bbox": {
                            "x": int(bbox_parts[0]),
                            "y": int(bbox_parts[1]),
                            "width": int(bbox_parts[2]),
                            "height": int(bbox_parts[3]),
                        },
                    })
        
        return OCRResult(
            text="\n".join(full_text),
            confidence=1.0,
            boxes=boxes,
            language=data.get("language"),
            orientation=data.get("orientation"),
            metadata={"provider": "azure"},
        )

    def extract_structured_data(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        schema: Optional[Dict[str, Any]] = None,
        prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Extract structured data from an image using OCR + LLM.
        
        Args:
            image: Input image
            schema: JSON schema for expected output structure
            prompt: Custom prompt for extraction
            
        Returns:
            Structured data extracted from the image
        """
        # First extract raw text
        ocr_result = self.extract_text(image, return_boxes=True)
        
        # Handle both string and OCRResult types
        if isinstance(ocr_result, str):
            raw_text = ocr_result
            confidence = 1.0
            boxes: List[Dict[str, Any]] = []
        else:
            raw_text = ocr_result.text
            confidence = ocr_result.confidence
            boxes = ocr_result.boxes or []
        
        if not self.llm_manager:
            # Return basic structure without LLM
            return {
                "raw_text": raw_text,
                "confidence": confidence,
                "boxes": boxes,
            }
        
        # Use LLM to structure the data
        extraction_prompt = prompt or self._build_extraction_prompt(raw_text, schema)
        
        response = self.llm_manager.query(extraction_prompt)
        
        try:
            # Try to parse as JSON
            structured_data = json.loads(response)
        except json.JSONDecodeError:
            # Return raw response
            structured_data = {"extracted": response}
        
        return {
            "raw_text": raw_text,
            "structured_data": structured_data,
            "confidence": confidence,
        }

    def _build_extraction_prompt(
        self, text: str, schema: Optional[Dict[str, Any]]
    ) -> str:
        """Build a prompt for structured data extraction."""
        base_prompt = f"""Extract structured information from the following text that was obtained via OCR.

OCR Text:
{text}

"""
        
        if schema:
            base_prompt += f"""
Extract the data according to this JSON schema:
{json.dumps(schema, indent=2)}

Return the extracted data as valid JSON.
"""
        else:
            base_prompt += """
Identify and extract any structured information (names, dates, numbers, addresses, etc.)
Return the extracted data as a JSON object with appropriate field names.
"""
        
        return base_prompt

    def process_document(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        document_type: str = "auto",
    ) -> Dict[str, Any]:
        """Process a document image with type-specific extraction.
        
        Args:
            image: Input image
            document_type: Type of document (invoice, receipt, id_card, form, auto)
            
        Returns:
            Extracted document data
        """
        # Extract text with boxes
        ocr_result = self.extract_text(image, return_boxes=True)
        
        # Document-specific schemas
        schemas = {
            "invoice": {
                "type": "object",
                "properties": {
                    "invoice_number": {"type": "string"},
                    "date": {"type": "string"},
                    "due_date": {"type": "string"},
                    "vendor": {"type": "string"},
                    "total": {"type": "number"},
                    "items": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "description": {"type": "string"},
                                "quantity": {"type": "number"},
                                "price": {"type": "number"},
                            },
                        },
                    },
                },
            },
            "receipt": {
                "type": "object",
                "properties": {
                    "store_name": {"type": "string"},
                    "date": {"type": "string"},
                    "total": {"type": "number"},
                    "items": {"type": "array"},
                    "payment_method": {"type": "string"},
                },
            },
            "id_card": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "id_number": {"type": "string"},
                    "birth_date": {"type": "string"},
                    "expiry_date": {"type": "string"},
                    "nationality": {"type": "string"},
                },
            },
        }
        
        schema = schemas.get(document_type)
        
        return self.extract_structured_data(image, schema=schema)

    def get_provider_info(self) -> Dict[str, Any]:
        """Get information about the current OCR provider."""
        return {
            "provider": self.provider,
            "languages": self.languages,
            "model_id": self.model_id,
            "device": self.device,
            "preprocessing": self.preprocessing,
            "available": self._available,
        }
