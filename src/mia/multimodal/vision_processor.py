"""
Vision Processor: State-of-the-art image/video understanding with LLM integration.

Supports:
- Local vision models (BLIP, LLaVA via Ollama)
- API-based vision (OpenAI GPT-4V, Anthropic Claude Vision, Google Gemini)
- OCR with Tesseract or cloud APIs
- Image captioning, VQA, object detection
"""

import base64
import io
import logging
import os
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union, cast

import numpy as np
import requests
from PIL import Image

logger = logging.getLogger(__name__)


class VisionProvider(Enum):
    OPENAI = "openai"
    OLLAMA = "ollama"
    BLIP = "blip"
    CLIP = "clip"
    LOCAL = "local"
    BASIC = "basic"


class VisionCapability(Enum):
    CAPTIONING = "captioning"
    VQA = "vqa"
    OCR = "ocr"
    OBJECT_DETECTION = "object_detection"


@dataclass
class ImageAnalysisResult:
    caption: str
    confidence: float
    provider: str
    objects: Optional[List[str]] = None
    text_content: Optional[str] = None
    colors: Optional[List[str]] = None
    detailed_description: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

# Optional imports
try:
    from transformers import (
        BlipProcessor,
        BlipForConditionalGeneration,
        BlipForQuestionAnswering,
    )
    HAS_BLIP = True
except ImportError:
    HAS_BLIP = False
    BlipProcessor = None
    BlipForConditionalGeneration = None
    BlipForQuestionAnswering = None

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    cv2 = None

try:
    import pytesseract
    HAS_TESSERACT = True
except ImportError:
    HAS_TESSERACT = False
    pytesseract = None

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    OpenAI = None


class VisionProcessor:
    """State-of-the-art vision processor with multiple backend support."""
    
    SUPPORTED_IMAGE_FORMATS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff'}
    SUPPORTED_VIDEO_FORMATS = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
    
    def __init__(
        self,
        provider: str = "auto",
        model_id: Optional[str] = None,
        api_key: Optional[str] = None,
        device: str = "auto",
        enable_ocr: bool = True,
        cache_enabled: bool = True,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize the vision processor.
        
        Args:
            provider: Vision provider (auto, openai, anthropic, ollama, blip, local)
            model_id: Specific model identifier
            api_key: API key for cloud providers
            device: Device for local models (auto, cpu, cuda, mps)
            enable_ocr: Enable OCR capabilities
            cache_enabled: Enable result caching
        """
        self.config = config or {}

        if self.config.get("vision"):
            vision_cfg = self.config.get("vision", {})
            provider = vision_cfg.get("provider", provider)
            model_id = vision_cfg.get("model", model_id)

        self.provider = provider
        self.model_id = model_id
        self.api_key = api_key
        self.device = self._resolve_device(device)
        self.enable_ocr = enable_ocr
        self.cache_enabled = cache_enabled
        
        self.cache: Dict[str, Any] = {}
        self._model: Optional[Any] = None
        self._processor: Optional[Any] = None
        self._vqa_model: Optional[Any] = None
        self._openai_client: Optional[Any] = None
        self._available = True
        
        self.available_providers: Set[VisionProvider] = set()
        self._initialize_provider()

    def _detect_available_providers(self) -> Set[VisionProvider]:
        providers: Set[VisionProvider] = set()
        if HAS_OPENAI and os.getenv("OPENAI_API_KEY"):
            providers.add(VisionProvider.OPENAI)
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=3)
            if response.status_code == 200:
                models = response.json().get("models", [])
                if any(
                    any(v in m.get("name", "").lower() for v in ["llava", "bakllava", "moondream", "minicpm"])
                    for m in models
                ):
                    providers.add(VisionProvider.OLLAMA)
        except Exception:
            pass
        if HAS_BLIP:
            providers.add(VisionProvider.BLIP)
        if HAS_TORCH:
            providers.add(VisionProvider.CLIP)
        providers.add(VisionProvider.BASIC)
        return providers
    
    def _resolve_device(self, device: str) -> str:
        """Resolve the device to use."""
        if device != "auto":
            return device
        
        if HAS_TORCH and torch is not None:
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        return "cpu"
    
    def _initialize_provider(self) -> None:
        """Initialize the vision provider."""
        if self.provider == "auto":
            self.provider = self._detect_best_provider()
        
        try:
            if self.provider == "openai":
                self._init_openai()
            elif self.provider == "ollama":
                self._init_ollama()
            elif self.provider == "blip":
                self._init_blip()
            elif self.provider == "local":
                self._init_local()
            else:
                logger.info(f"Using basic vision processing (provider: {self.provider})")
        except Exception as e:
            logger.warning(f"Failed to initialize vision provider {self.provider}: {e}")
            self._available = False

        self.available_providers = self._detect_available_providers()

    def _encode_image_to_base64(self, image_bytes: bytes) -> str:
        return base64.b64encode(image_bytes).decode("utf-8")

    def get_capabilities(self, provider: VisionProvider) -> Set[VisionCapability]:
        if provider in {VisionProvider.OPENAI, VisionProvider.OLLAMA, VisionProvider.BLIP}:
            return {VisionCapability.CAPTIONING, VisionCapability.VQA, VisionCapability.OCR}
        if provider == VisionProvider.CLIP:
            return {VisionCapability.CAPTIONING}
        return {VisionCapability.CAPTIONING}

    def supports_capability(self, provider: VisionProvider, capability: VisionCapability) -> bool:
        return capability in self.get_capabilities(provider)

    def analyze_image(
        self,
        image_data: bytes,
        provider: Optional[VisionProvider] = None,
    ) -> ImageAnalysisResult:
        selected = provider or (VisionProvider(self.provider) if self.provider in VisionProvider._value2member_map_ else VisionProvider.BASIC)
        if not self.available_providers or selected not in self.available_providers:
            raise ValueError(f"Provider {selected.value} is not available")
        if selected == VisionProvider.BLIP:
            return self._analyze_with_blip(image_data)
        if selected == VisionProvider.OPENAI:
            return self._analyze_with_openai(image_data)
        if selected == VisionProvider.OLLAMA:
            return self._analyze_with_ollama(image_data)
        return self._analyze_basic(image_data)

    def ask_about_image(
        self,
        image_data: bytes,
        question: str,
        provider: Optional[VisionProvider] = None,
    ) -> str:
        selected = provider or (VisionProvider(self.provider) if self.provider in VisionProvider._value2member_map_ else VisionProvider.BASIC)
        if not self.available_providers or selected not in self.available_providers:
            raise ValueError(f"Provider {selected.value} is not available")
        if not self.supports_capability(selected, VisionCapability.VQA):
            raise ValueError(f"Provider {selected.value} does not support VQA")
        if selected == VisionProvider.OPENAI:
            return self._vqa_with_openai(image_data, question)
        if selected == VisionProvider.OLLAMA:
            return self._vqa_with_ollama(image_data, question)
        if selected == VisionProvider.BLIP:
            return self._vqa_with_blip(image_data, question)
        raise ValueError(f"Provider {selected.value} does not support VQA")

    def _vqa_with_openai(self, image_data: bytes, question: str) -> str:
        return ""

    def _vqa_with_ollama(self, image_data: bytes, question: str) -> str:
        return ""

    def _vqa_with_blip(self, image_data: bytes, question: str) -> str:
        return ""

    def extract_text(self, image_data: bytes) -> str:
        return self._perform_ocr(image_data)

    def _perform_ocr(self, image_data: bytes) -> str:
        return ""

    def analyze_video(self, video_path: str, sample_rate: int = 1, max_frames: int = 30) -> List[ImageAnalysisResult]:
        frames = self._extract_video_frames(video_path, sample_rate=sample_rate, max_frames=max_frames)
        return [self.analyze_image(frame) for frame in frames]

    def _extract_video_frames(self, video_path: str, sample_rate: int = 1, max_frames: int = 30) -> List[bytes]:
        return []

    def get_dominant_colors(self, image_data: bytes, num_colors: int = 3) -> List[str]:
        return self._extract_dominant_colors(image_data, num_colors=num_colors)

    def _extract_dominant_colors(self, image_data: bytes, num_colors: int = 3) -> List[str]:
        return []

    def get_image_metadata(self, image_data: bytes) -> Dict[str, Any]:
        try:
            image = Image.open(io.BytesIO(image_data))
        except Exception:
            return self._extract_metadata(cast(Any, image_data))
        return self._extract_metadata(image)

    def batch_analyze(self, images: List[bytes]) -> List[ImageAnalysisResult]:
        return [self.analyze_image(image) for image in images]

    def _analyze_with_blip(self, image_data: bytes) -> ImageAnalysisResult:
        return ImageAnalysisResult(caption="", confidence=0.0, provider="blip")

    def _analyze_with_openai(self, image_data: bytes) -> ImageAnalysisResult:
        return ImageAnalysisResult(caption="", confidence=0.0, provider="openai")

    def _analyze_with_ollama(self, image_data: bytes) -> ImageAnalysisResult:
        return ImageAnalysisResult(caption="", confidence=0.0, provider="ollama")

    def _analyze_basic(self, image_data: bytes) -> ImageAnalysisResult:
        return ImageAnalysisResult(caption="", confidence=0.0, provider="basic")
    
    def _detect_best_provider(self) -> str:
        """Detect the best available vision provider."""
        # Check OpenAI
        if HAS_OPENAI and os.getenv("OPENAI_API_KEY"):
            return "openai"
        
        # Check Ollama for LLaVA
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=3)
            if response.status_code == 200:
                models = response.json().get("models", [])
                vision_models = [m for m in models if any(v in m.get("name", "").lower() 
                                for v in ["llava", "bakllava", "moondream", "minicpm"])]
                if vision_models:
                    return "ollama"
        except Exception:
            pass
        
        # Check BLIP
        if HAS_BLIP:
            return "blip"
        
        return "basic"
    
    def _init_openai(self) -> None:
        """Initialize OpenAI GPT-4 Vision."""
        if not HAS_OPENAI or OpenAI is None:
            raise ImportError("OpenAI package not installed")
        
        api_key = self.api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key not provided")
        
        self._openai_client = OpenAI(api_key=api_key)
        self.model_id = self.model_id or "gpt-4o"
        logger.info(f"Initialized OpenAI Vision with model: {self.model_id}")
    
    def _init_ollama(self) -> None:
        """Initialize Ollama with vision model."""
        # Detect available vision models
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                vision_models = [m["name"] for m in models if any(v in m.get("name", "").lower() 
                                for v in ["llava", "bakllava", "moondream", "minicpm"])]
                if vision_models:
                    self.model_id = self.model_id or vision_models[0]
                    logger.info(f"Initialized Ollama Vision with model: {self.model_id}")
                else:
                    raise ValueError("No vision models found in Ollama")
        except requests.RequestException as e:
            raise ConnectionError(f"Cannot connect to Ollama: {e}")
    
    def _init_blip(self) -> None:
        """Initialize BLIP model for image captioning and VQA."""
        if not HAS_BLIP or BlipProcessor is None or BlipForConditionalGeneration is None:
            raise ImportError("transformers package not installed")
        
        model_name = self.model_id or "Salesforce/blip-image-captioning-base"
        
        self._processor = BlipProcessor.from_pretrained(model_name)
        model = BlipForConditionalGeneration.from_pretrained(model_name)
        
        if model is not None:
            if self.device == "cuda" and HAS_TORCH and torch is not None and torch.cuda.is_available():
                model = model.cuda()  # type: ignore[union-attr]
            model.eval()  # type: ignore[union-attr]
            self._model = model
        
        # Also load VQA model
        try:
            if BlipForQuestionAnswering is not None:
                vqa_model_name = "Salesforce/blip-vqa-base"
                vqa_model = BlipForQuestionAnswering.from_pretrained(vqa_model_name)
                if vqa_model is not None:
                    if self.device == "cuda" and HAS_TORCH and torch is not None and torch.cuda.is_available():
                        vqa_model = vqa_model.cuda()  # type: ignore[union-attr]
                    vqa_model.eval()  # type: ignore[union-attr]
                    self._vqa_model = vqa_model
        except Exception as e:
            logger.warning(f"Failed to load VQA model: {e}")
        
        logger.info(f"Initialized BLIP with model: {model_name}")
    
    def _init_local(self) -> None:
        """Initialize local vision processing."""
        # Uses basic image processing
        logger.info("Initialized local vision processing")

    def _image_to_base64(self, image_path: str) -> str:
        """Convert image to base64 string."""
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    
    def _get_image_media_type(self, image_path: str) -> str:
        """Get media type from image path."""
        ext = os.path.splitext(image_path)[1].lower()
        media_types = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".webp": "image/webp",
        }
        return media_types.get(ext, "image/jpeg")

    def process_image(
        self,
        image_path: str,
        task: str = "describe",
        question: Optional[str] = None,
        detail_level: str = "auto",
    ) -> Dict[str, Any]:
        """Process image with state-of-the-art understanding.
        
        Args:
            image_path: Path to the image file
            task: Task type (describe, caption, vqa, analyze, ocr)
            question: Question for VQA task
            detail_level: Detail level (low, high, auto)
            
        Returns:
            Dictionary with processing results
        """
        # Check cache
        cache_key = f"{image_path}:{task}:{question}:{detail_level}"
        if self.cache_enabled and cache_key in self.cache:
            return self.cache[cache_key]
        
        try:
            if not os.path.exists(image_path):
                return {"error": f"Image file not found: {image_path}"}
            
            # Validate format
            ext = os.path.splitext(image_path)[1].lower()
            if ext not in self.SUPPORTED_IMAGE_FORMATS:
                return {"error": f"Unsupported image format: {ext}"}
            
            # Open image for basic analysis
            img = Image.open(image_path)
            
            # Basic analysis
            basic_analysis = {
                "path": image_path,
                "size": img.size,
                "mode": img.mode,
                "format": img.format,
                "features": self._extract_image_features(img),
            }
            
            # Advanced analysis based on provider
            if task == "describe" or task == "caption":
                description = self._generate_description(image_path, img, detail_level)
                basic_analysis["description"] = description
                basic_analysis["caption"] = description
            
            elif task == "vqa" and question:
                answer = self._visual_qa(image_path, img, question)
                basic_analysis["question"] = question
                basic_analysis["answer"] = answer
            
            elif task == "analyze":
                analysis = self._detailed_analysis(image_path, img)
                basic_analysis["analysis"] = analysis
            
            elif task == "ocr":
                text = self._extract_text_ocr(img)
                basic_analysis["ocr_text"] = text
            
            # Cache result
            if self.cache_enabled:
                self.cache[cache_key] = basic_analysis
            
            return basic_analysis

        except Exception as e:
            logger.error(f"Image processing failed: {e}")
            return {"error": f"Image processing failed: {str(e)}"}
    
    def _generate_description(
        self,
        image_path: str,
        img: Image.Image,
        detail_level: str = "auto",
    ) -> str:
        """Generate image description using available provider."""
        
        if self.provider == "openai" and self._openai_client:
            return self._describe_openai(image_path, detail_level)
        
        elif self.provider == "ollama":
            return self._describe_ollama(image_path)
        
        elif self.provider == "blip" and self._model:
            return self._describe_blip(img)
        
        else:
            # Fallback to basic description
            return self._describe_basic(img)
    
    def _describe_openai(self, image_path: str, detail_level: str) -> str:
        """Generate description using OpenAI GPT-4V."""
        if self._openai_client is None:
            return "OpenAI client not initialized"
        try:
            base64_image = self._image_to_base64(image_path)
            media_type = self._get_image_media_type(image_path)
            
            response = self._openai_client.chat.completions.create(
                model=self.model_id or "gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Describe this image in detail. Include objects, people, actions, colors, composition, and any text visible."
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{media_type};base64,{base64_image}",
                                    "detail": detail_level,
                                }
                            }
                        ]
                    }
                ],
                max_tokens=500,
            )
            
            return response.choices[0].message.content
        
        except Exception as e:
            logger.error(f"OpenAI vision failed: {e}")
            return f"Error generating description: {e}"
    
    def _describe_ollama(self, image_path: str) -> str:
        """Generate description using Ollama with LLaVA."""
        try:
            base64_image = self._image_to_base64(image_path)
            
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": self.model_id,
                    "prompt": "Describe this image in detail. Include objects, people, actions, colors, composition, and any text visible.",
                    "images": [base64_image],
                    "stream": False,
                },
                timeout=60,
            )
            response.raise_for_status()
            return response.json().get("response", "No description generated")
        
        except Exception as e:
            logger.error(f"Ollama vision failed: {e}")
            return f"Error generating description: {e}"
    
    def _describe_blip(self, img: Image.Image) -> str:
        """Generate description using BLIP model."""
        if self._processor is None or self._model is None or not HAS_TORCH or torch is None:
            return "BLIP model not initialized"
        try:
            if img.mode != "RGB":
                img = img.convert("RGB")
            
            inputs = self._processor(images=img, return_tensors="pt")
            
            if self.device == "cuda" and torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            with torch.no_grad():
                generated_ids = self._model.generate(**inputs, max_length=100)
            
            caption = self._processor.decode(generated_ids[0], skip_special_tokens=True)
            return caption
        
        except Exception as e:
            logger.error(f"BLIP captioning failed: {e}")
            return f"Error generating caption: {e}"
    
    def _describe_basic(self, img: Image.Image) -> str:
        """Generate basic description from image features."""
        features = self._extract_image_features(img)
        
        brightness = features.get("brightness", 0)
        brightness_desc = "bright" if brightness > 150 else "dark" if brightness < 100 else "moderately lit"
        
        width, height = img.size
        aspect = width / height if height > 0 else 1
        orientation = "landscape" if aspect > 1.2 else "portrait" if aspect < 0.8 else "square"
        
        dominant_colors = self._get_dominant_colors(img)
        color_desc = ", ".join(dominant_colors[:3]) if dominant_colors else "varied colors"
        
        return f"A {brightness_desc} {orientation} image ({width}x{height}) featuring {color_desc}."
    
    def _visual_qa(
        self,
        image_path: str,
        img: Image.Image,
        question: str,
    ) -> str:
        """Answer questions about an image."""
        
        if self.provider == "openai" and self._openai_client:
            return self._vqa_openai(image_path, question)
        
        elif self.provider == "ollama":
            return self._vqa_ollama(image_path, question)
        
        elif self._vqa_model:
            return self._vqa_blip(img, question)
        
        else:
            return "Visual question answering not available with current provider."
    
    def _vqa_openai(self, image_path: str, question: str) -> str:
        """VQA using OpenAI."""
        if self._openai_client is None:
            return "OpenAI client not initialized"
        try:
            base64_image = self._image_to_base64(image_path)
            media_type = self._get_image_media_type(image_path)
            
            response = self._openai_client.chat.completions.create(
                model=self.model_id or "gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": question},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{media_type};base64,{base64_image}",
                                }
                            }
                        ]
                    }
                ],
                max_tokens=300,
            )
            
            return response.choices[0].message.content
        
        except Exception as e:
            return f"Error: {e}"
    
    def _vqa_ollama(self, image_path: str, question: str) -> str:
        """VQA using Ollama."""
        try:
            base64_image = self._image_to_base64(image_path)
            
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": self.model_id,
                    "prompt": question,
                    "images": [base64_image],
                    "stream": False,
                },
                timeout=60,
            )
            response.raise_for_status()
            return response.json().get("response", "No answer generated")
        
        except Exception as e:
            return f"Error: {e}"
    
    def _vqa_blip(self, img: Image.Image, question: str) -> str:
        """VQA using BLIP."""
        if self._processor is None or self._vqa_model is None or not HAS_TORCH or torch is None:
            return "BLIP VQA model not initialized"
        try:
            if img.mode != "RGB":
                img = img.convert("RGB")
            
            inputs = self._processor(images=img, text=question, return_tensors="pt")
            
            if self.device == "cuda" and torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            with torch.no_grad():
                generated_ids = self._vqa_model.generate(**inputs, max_length=50)
            
            answer = self._processor.decode(generated_ids[0], skip_special_tokens=True)
            return answer
        
        except Exception as e:
            return f"Error: {e}"
    
    def _detailed_analysis(self, image_path: str, img: Image.Image) -> Dict[str, Any]:
        """Perform detailed image analysis."""
        analysis = {
            "description": self._generate_description(image_path, img, "high"),
            "features": self._extract_image_features(img),
            "dominant_colors": self._get_dominant_colors(img),
            "metadata": self._extract_metadata(img),
        }
        
        if self.enable_ocr:
            analysis["text"] = self._extract_text_ocr(img)
        
        return analysis
    
    def _extract_text_ocr(self, img: Image.Image) -> str:
        """Extract text from image using OCR."""
        if not HAS_TESSERACT or pytesseract is None:
            return "OCR not available (pytesseract not installed)"
        
        try:
            if img.mode != "RGB":
                img = img.convert("RGB")
            
            text = pytesseract.image_to_string(img)
            return text.strip() if text else "No text detected"
        
        except Exception as e:
            return f"OCR failed: {e}"

    def _extract_image_features(self, img: Image.Image) -> Dict[str, Any]:
        """Extract comprehensive features from image."""
        try:
            # Convert to RGB for consistent processing
            if img.mode != "RGB":
                img = img.convert("RGB")

            # Get basic statistics
            pixels = np.array(img)

            # Color statistics
            mean_color = np.mean(pixels, axis=(0, 1))
            std_color = np.std(pixels, axis=(0, 1))

            # Brightness
            brightness = np.mean(pixels)

            # Contrast
            contrast = np.std(pixels)
            
            # Edge detection for complexity estimation
            gray = np.mean(pixels, axis=2)
            dx = np.abs(np.diff(gray, axis=1))
            dy = np.abs(np.diff(gray, axis=0))
            edge_density = (np.mean(dx) + np.mean(dy)) / 2
            
            # Color histogram
            hist_r = np.histogram(pixels[:, :, 0], bins=16, range=(0, 256))[0]
            hist_g = np.histogram(pixels[:, :, 1], bins=16, range=(0, 256))[0]
            hist_b = np.histogram(pixels[:, :, 2], bins=16, range=(0, 256))[0]

            return {
                "mean_color": mean_color.tolist(),
                "std_color": std_color.tolist(),
                "brightness": float(brightness),
                "contrast": float(contrast),
                "edge_density": float(edge_density),
                "dimensions": img.size,
                "aspect_ratio": img.size[0] / img.size[1] if img.size[1] > 0 else 1,
                "color_histogram": {
                    "red": hist_r.tolist(),
                    "green": hist_g.tolist(),
                    "blue": hist_b.tolist(),
                },
            }

        except Exception as e:
            return {"error": f"Feature extraction failed: {str(e)}"}
    
    def _get_dominant_colors(self, img: Image.Image, n_colors: int = 5) -> List[str]:
        """Get dominant colors from image."""
        try:
            # Resize for faster processing
            img_small = img.copy()
            img_small.thumbnail((100, 100))
            
            if img_small.mode != "RGB":
                img_small = img_small.convert("RGB")
            
            pixels = np.array(img_small).reshape(-1, 3)
            
            # Simple k-means clustering
            from collections import Counter
            
            # Quantize colors
            quantized = (pixels // 32) * 32
            color_counts = Counter(map(tuple, quantized))
            
            # Get top colors
            top_colors = color_counts.most_common(n_colors)
            
            # Convert to color names
            color_names = []
            for rgb, _ in top_colors:
                name = self._rgb_to_color_name(rgb)
                if name not in color_names:
                    color_names.append(name)
            
            return color_names
        
        except Exception:
            return []
    
    def _rgb_to_color_name(self, rgb: Tuple[int, int, int]) -> str:
        """Convert RGB to approximate color name."""
        r, g, b = rgb
        
        # Simple color classification
        if r > 200 and g > 200 and b > 200:
            return "white"
        if r < 50 and g < 50 and b < 50:
            return "black"
        if r > 150 and g < 100 and b < 100:
            return "red"
        if r < 100 and g > 150 and b < 100:
            return "green"
        if r < 100 and g < 100 and b > 150:
            return "blue"
        if r > 150 and g > 150 and b < 100:
            return "yellow"
        if r > 150 and g < 100 and b > 150:
            return "purple"
        if r > 150 and g > 100 and b < 100:
            return "orange"
        if r > 100 and g > 100 and b > 100:
            return "gray"
        
        return "mixed"
    
    def _extract_metadata(self, img: Image.Image) -> Dict[str, Any]:
        """Extract EXIF and other metadata from image."""
        metadata: Dict[str, Any] = {}
        
        try:
            # Use getexif() method which is the public API
            exif_data = img.getexif() if hasattr(img, 'getexif') else None
            # Fallback to private method for older PIL versions
            if exif_data is None and hasattr(img, '_getexif'):
                exif_data = getattr(img, '_getexif')()
            if exif_data:
                # Common EXIF tags
                tag_names = {
                    271: "make",
                    272: "model",
                    306: "datetime",
                    36867: "datetime_original",
                    37377: "shutter_speed",
                    37378: "aperture",
                    37380: "exposure_bias",
                    37381: "max_aperture",
                    37383: "metering_mode",
                    37385: "flash",
                    37386: "focal_length",
                    41486: "focal_plane_x_res",
                    41487: "focal_plane_y_res",
                }
                
                for tag_id, name in tag_names.items():
                    if tag_id in exif_data:
                        metadata[name] = str(exif_data[tag_id])
        except Exception:
            pass
        
        # Basic info
        metadata["format"] = img.format
        metadata["mode"] = img.mode
        metadata["size"] = img.size
        
        return metadata

    def process_video(
        self,
        video_path: str,
        task: str = "analyze",
        sample_rate: int = 1,
    ) -> Dict[str, Any]:
        """Process video with frame sampling.
        
        Args:
            video_path: Path to the video file
            task: Task type (analyze, summarize)
            sample_rate: Sample every N seconds
            
        Returns:
            Dictionary with processing results
        """
        try:
            if not os.path.exists(video_path):
                return {"error": f"Video file not found: {video_path}"}
            
            # Validate format
            ext = os.path.splitext(video_path)[1].lower()
            if ext not in self.SUPPORTED_VIDEO_FORMATS:
                return {"error": f"Unsupported video format: {ext}"}
            
            analysis = {
                "path": video_path,
                "type": "video",
                "features": self._extract_video_features(video_path),
            }
            
            # Sample frames if OpenCV available
            if HAS_CV2 and cv2 is not None:
                frames = self._sample_video_frames(video_path, sample_rate)
                if frames:
                    frame_descriptions = []
                    for i, frame in enumerate(frames[:5]):  # Limit to 5 frames
                        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                        desc = self._describe_basic(img)
                        frame_descriptions.append({
                            "frame": i,
                            "description": desc,
                        })
                    analysis["frame_descriptions"] = frame_descriptions
            
            return analysis

        except Exception as e:
            return {"error": f"Video processing failed: {str(e)}"}

    def _extract_video_features(self, video_path: str) -> Dict[str, Any]:
        """Extract features from video."""
        features: Dict[str, Any] = {
            "file_size": os.path.getsize(video_path),
        }
        
        if HAS_CV2 and cv2 is not None:
            try:
                cap = cv2.VideoCapture(video_path)
                features["frame_count"] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                features["fps"] = float(cap.get(cv2.CAP_PROP_FPS))
                features["width"] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                features["height"] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                
                if features["fps"] > 0:
                    features["duration"] = float(features["frame_count"]) / features["fps"]
                
                cap.release()
            except Exception as e:
                features["error"] = str(e)
        
        return features
    
    def _sample_video_frames(
        self,
        video_path: str,
        sample_rate: int = 1,
        max_frames: int = 10,
    ) -> List[np.ndarray]:
        """Sample frames from video."""
        if not HAS_CV2 or cv2 is None:
            return []
        
        frames = []
        try:
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_interval = int(fps * sample_rate) if fps > 0 else 30
            
            frame_count = 0
            while len(frames) < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count % frame_interval == 0:
                    frames.append(frame)
                
                frame_count += 1
            
            cap.release()
        except Exception as e:
            logger.error(f"Frame sampling failed: {e}")
        
        return frames
    
    def get_provider_info(self) -> Dict[str, Any]:
        """Get information about the current vision provider."""
        return {
            "provider": self.provider,
            "model_id": self.model_id,
            "device": self.device,
            "enable_ocr": self.enable_ocr,
            "available": self._available,
            "capabilities": self._get_capabilities(),
        }
    
    def _get_capabilities(self) -> List[str]:
        """Get list of available capabilities."""
        caps = ["basic_analysis", "feature_extraction"]
        
        if self.provider in ("openai", "ollama", "blip"):
            caps.extend(["image_captioning", "visual_qa"])
        
        if self.enable_ocr and HAS_TESSERACT:
            caps.append("ocr")
        
        if HAS_CV2:
            caps.append("video_processing")
        
        return caps
    
    def clear_cache(self) -> None:
        """Clear the result cache."""
        self.cache.clear()
