"""
Comprehensive tests for the Vision Processor module.
Tests multi-provider vision, VQA, OCR, and image analysis.
"""

import os
import sys
import base64
import unittest
from unittest.mock import MagicMock, patch, AsyncMock
from io import BytesIO

# Add src directory to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
src_dir = os.path.join(project_root, "src")
sys.path.insert(0, src_dir)

from mia.multimodal.vision_processor import (  # type: ignore[import-not-found]
    VisionProcessor,
    VisionProvider,
    VisionCapability,
    ImageAnalysisResult,
)


class TestVisionProvider(unittest.TestCase):
    """Tests for VisionProvider enum."""
    
    def test_provider_values(self):
        """Test provider enum values exist."""
        self.assertEqual(VisionProvider.OPENAI.value, "openai")
        self.assertEqual(VisionProvider.OLLAMA.value, "ollama")
        self.assertEqual(VisionProvider.BLIP.value, "blip")
        self.assertEqual(VisionProvider.CLIP.value, "clip")


class TestVisionCapability(unittest.TestCase):
    """Tests for VisionCapability enum."""
    
    def test_capability_values(self):
        """Test capability enum values exist."""
        self.assertEqual(VisionCapability.CAPTIONING.value, "captioning")
        self.assertEqual(VisionCapability.VQA.value, "vqa")
        self.assertEqual(VisionCapability.OCR.value, "ocr")
        self.assertEqual(VisionCapability.OBJECT_DETECTION.value, "object_detection")


class TestImageAnalysisResult(unittest.TestCase):
    """Tests for ImageAnalysisResult dataclass."""
    
    def test_basic_creation(self):
        """Test creating analysis result."""
        result = ImageAnalysisResult(
            caption="A dog playing",
            confidence=0.95,
            provider="openai",
        )
        
        self.assertEqual(result.caption, "A dog playing")
        self.assertEqual(result.confidence, 0.95)
        self.assertEqual(result.provider, "openai")
    
    def test_optional_fields(self):
        """Test optional fields."""
        result = ImageAnalysisResult(
            caption="Test",
            confidence=0.8,
            provider="blip",
            objects=["car", "tree"],
            text_content="Some OCR text",
            colors=["red", "blue"],
            detailed_description="A detailed description here.",
            metadata={"width": 1920, "height": 1080},
        )
        
        self.assertEqual(result.objects, ["car", "tree"])
        self.assertEqual(result.text_content, "Some OCR text")
        self.assertEqual(result.colors, ["red", "blue"])
        self.assertEqual(result.detailed_description, "A detailed description here.")
        self.assertEqual(result.metadata["width"], 1920)
    
    def test_default_optional_fields(self):
        """Test default values for optional fields."""
        result = ImageAnalysisResult(
            caption="Test",
            confidence=0.5,
            provider="test",
        )
        
        self.assertIsNone(result.objects)
        self.assertIsNone(result.text_content)
        self.assertIsNone(result.colors)
        self.assertIsNone(result.detailed_description)
        self.assertIsNone(result.metadata)


class TestVisionProcessor(unittest.TestCase):
    """Tests for VisionProcessor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_config = {
            "vision": {
                "enabled": True,
                "provider": "blip",
                "openai_model": "gpt-4-vision-preview",
                "ollama_model": "llava",
                "blip_model": "Salesforce/blip-image-captioning-base",
            }
        }
    
    @patch("mia.multimodal.vision_processor.VisionProcessor._detect_available_providers")
    def test_initialization(self, mock_detect):
        """Test VisionProcessor initialization."""
        mock_detect.return_value = {VisionProvider.BLIP}
        
        processor = VisionProcessor(config=self.mock_config)
        
        self.assertIsNotNone(processor)
        self.assertEqual(processor.config, self.mock_config)
    
    @patch("mia.multimodal.vision_processor.VisionProcessor._detect_available_providers")
    def test_available_providers(self, mock_detect):
        """Test provider detection."""
        mock_detect.return_value = {VisionProvider.BLIP, VisionProvider.OPENAI}
        
        processor = VisionProcessor(config=self.mock_config)
        
        self.assertIn(VisionProvider.BLIP, processor.available_providers)
    
    def test_encode_image_base64(self):
        """Test base64 image encoding."""
        # Create a simple test image (1x1 pixel PNG)
        test_image_bytes = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82'
        
        with patch("mia.multimodal.vision_processor.VisionProcessor._detect_available_providers"):
            processor = VisionProcessor(config=self.mock_config)
            
            # Test encoding
            encoded = processor._encode_image_to_base64(test_image_bytes)
            
            self.assertIsInstance(encoded, str)
            # Should be valid base64
            decoded = base64.b64decode(encoded)
            self.assertEqual(decoded, test_image_bytes)
    
    @patch("mia.multimodal.vision_processor.VisionProcessor._detect_available_providers")
    def test_get_capabilities(self, mock_detect):
        """Test getting provider capabilities."""
        mock_detect.return_value = {VisionProvider.OPENAI}
        
        processor = VisionProcessor(config=self.mock_config)
        
        capabilities = processor.get_capabilities(VisionProvider.OPENAI)
        
        self.assertIn(VisionCapability.CAPTIONING, capabilities)
        self.assertIn(VisionCapability.VQA, capabilities)
    
    @patch("mia.multimodal.vision_processor.VisionProcessor._detect_available_providers")
    def test_supports_capability(self, mock_detect):
        """Test checking capability support."""
        mock_detect.return_value = {VisionProvider.OPENAI}
        
        processor = VisionProcessor(config=self.mock_config)
        
        self.assertTrue(processor.supports_capability(VisionProvider.OPENAI, VisionCapability.VQA))


class TestVisionProcessorAnalysis(unittest.TestCase):
    """Tests for image analysis methods."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_config = {
            "vision": {
                "enabled": True,
                "provider": "blip",
            }
        }
    
    @patch("mia.multimodal.vision_processor.VisionProcessor._detect_available_providers")
    @patch("mia.multimodal.vision_processor.VisionProcessor._analyze_with_blip")
    def test_analyze_image_with_blip(self, mock_blip, mock_detect):
        """Test image analysis with BLIP provider."""
        mock_detect.return_value = {VisionProvider.BLIP}
        mock_blip.return_value = ImageAnalysisResult(
            caption="A cat sitting on a couch",
            confidence=0.92,
            provider="blip",
        )
        
        processor = VisionProcessor(config=self.mock_config)
        
        result = processor.analyze_image(
            image_data=b"fake_image_data",
            provider=VisionProvider.BLIP,
        )
        
        self.assertEqual(result.caption, "A cat sitting on a couch")
        self.assertEqual(result.confidence, 0.92)
        mock_blip.assert_called_once()
    
    @patch("mia.multimodal.vision_processor.VisionProcessor._detect_available_providers")
    @patch("mia.multimodal.vision_processor.VisionProcessor._analyze_with_openai")
    def test_analyze_image_with_openai(self, mock_openai, mock_detect):
        """Test image analysis with OpenAI provider."""
        mock_detect.return_value = {VisionProvider.OPENAI}
        mock_openai.return_value = ImageAnalysisResult(
            caption="A beautiful sunset over the ocean",
            confidence=0.95,
            provider="openai",
            detailed_description="The image shows a vibrant sunset...",
        )
        
        processor = VisionProcessor(config=self.mock_config)
        
        result = processor.analyze_image(
            image_data=b"fake_image_data",
            provider=VisionProvider.OPENAI,
        )
        
        self.assertEqual(result.caption, "A beautiful sunset over the ocean")
        self.assertIsNotNone(result.detailed_description)
        mock_openai.assert_called_once()
    
    @patch("mia.multimodal.vision_processor.VisionProcessor._detect_available_providers")
    def test_analyze_image_invalid_provider(self, mock_detect):
        """Test analysis with unavailable provider raises error."""
        mock_detect.return_value = set()  # No providers available
        
        processor = VisionProcessor(config=self.mock_config)
        
        with self.assertRaises(ValueError):
            processor.analyze_image(
                image_data=b"fake_image",
                provider=VisionProvider.OPENAI,
            )


class TestVisualQuestionAnswering(unittest.TestCase):
    """Tests for VQA functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_config = {
            "vision": {
                "enabled": True,
                "provider": "openai",
            }
        }
    
    @patch("mia.multimodal.vision_processor.VisionProcessor._detect_available_providers")
    @patch("mia.multimodal.vision_processor.VisionProcessor._vqa_with_openai")
    def test_ask_about_image(self, mock_vqa, mock_detect):
        """Test visual question answering."""
        mock_detect.return_value = {VisionProvider.OPENAI}
        mock_vqa.return_value = "There are 3 people in the image."
        
        processor = VisionProcessor(config=self.mock_config)
        
        answer = processor.ask_about_image(
            image_data=b"fake_image_data",
            question="How many people are in the image?",
            provider=VisionProvider.OPENAI,
        )
        
        self.assertEqual(answer, "There are 3 people in the image.")
        mock_vqa.assert_called_once()
    
    @patch("mia.multimodal.vision_processor.VisionProcessor._detect_available_providers")
    def test_ask_about_image_no_vqa_support(self, mock_detect):
        """Test VQA with provider that doesn't support it."""
        mock_detect.return_value = {VisionProvider.CLIP}  # CLIP doesn't support VQA
        
        processor = VisionProcessor(config=self.mock_config)
        
        with self.assertRaises(ValueError):
            processor.ask_about_image(
                image_data=b"fake_image",
                question="What is this?",
                provider=VisionProvider.CLIP,
            )


class TestOCRFunctionality(unittest.TestCase):
    """Tests for OCR functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_config = {
            "vision": {
                "enabled": True,
                "provider": "blip",
                "ocr_enabled": True,
            }
        }
    
    @patch("mia.multimodal.vision_processor.VisionProcessor._detect_available_providers")
    @patch("mia.multimodal.vision_processor.VisionProcessor._perform_ocr")
    def test_extract_text(self, mock_ocr, mock_detect):
        """Test text extraction from image."""
        mock_detect.return_value = {VisionProvider.BLIP}
        mock_ocr.return_value = "Hello, World!\nThis is OCR text."
        
        processor = VisionProcessor(config=self.mock_config)
        
        text = processor.extract_text(image_data=b"fake_image_data")
        
        self.assertEqual(text, "Hello, World!\nThis is OCR text.")
        mock_ocr.assert_called_once()
    
    @patch("mia.multimodal.vision_processor.VisionProcessor._detect_available_providers")
    @patch("mia.multimodal.vision_processor.VisionProcessor._perform_ocr")
    def test_extract_text_empty(self, mock_ocr, mock_detect):
        """Test OCR with no text found."""
        mock_detect.return_value = {VisionProvider.BLIP}
        mock_ocr.return_value = ""
        
        processor = VisionProcessor(config=self.mock_config)
        
        text = processor.extract_text(image_data=b"fake_image_data")
        
        self.assertEqual(text, "")


class TestVideoProcessing(unittest.TestCase):
    """Tests for video processing functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_config = {
            "vision": {
                "enabled": True,
                "provider": "openai",
            }
        }
    
    @patch("mia.multimodal.vision_processor.VisionProcessor._detect_available_providers")
    @patch("mia.multimodal.vision_processor.VisionProcessor._extract_video_frames")
    @patch("mia.multimodal.vision_processor.VisionProcessor.analyze_image")
    def test_analyze_video(self, mock_analyze, mock_extract, mock_detect):
        """Test video analysis by frame sampling."""
        mock_detect.return_value = {VisionProvider.OPENAI}
        mock_extract.return_value = [b"frame1", b"frame2", b"frame3"]
        mock_analyze.return_value = ImageAnalysisResult(
            caption="A person walking",
            confidence=0.9,
            provider="openai",
        )
        
        processor = VisionProcessor(config=self.mock_config)
        
        results = processor.analyze_video(
            video_path="/path/to/video.mp4",
            sample_rate=1,  # 1 frame per second
            max_frames=5,
        )
        
        self.assertEqual(len(results), 3)
        self.assertEqual(mock_analyze.call_count, 3)


class TestColorExtraction(unittest.TestCase):
    """Tests for dominant color extraction."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_config = {
            "vision": {
                "enabled": True,
                "provider": "blip",
            }
        }
    
    @patch("mia.multimodal.vision_processor.VisionProcessor._detect_available_providers")
    @patch("mia.multimodal.vision_processor.VisionProcessor._extract_dominant_colors")
    def test_get_dominant_colors(self, mock_colors, mock_detect):
        """Test extracting dominant colors."""
        mock_detect.return_value = {VisionProvider.BLIP}
        mock_colors.return_value = ["#FF5733", "#33FF57", "#3357FF"]
        
        processor = VisionProcessor(config=self.mock_config)
        
        colors = processor.get_dominant_colors(
            image_data=b"fake_image_data",
            num_colors=3,
        )
        
        self.assertEqual(len(colors), 3)
        self.assertIn("#FF5733", colors)


class TestMetadataExtraction(unittest.TestCase):
    """Tests for image metadata extraction."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_config = {
            "vision": {
                "enabled": True,
                "provider": "blip",
            }
        }
    
    @patch("mia.multimodal.vision_processor.VisionProcessor._detect_available_providers")
    @patch("mia.multimodal.vision_processor.VisionProcessor._extract_metadata")
    def test_get_image_metadata(self, mock_meta, mock_detect):
        """Test extracting image metadata."""
        mock_detect.return_value = {VisionProvider.BLIP}
        mock_meta.return_value = {
            "width": 1920,
            "height": 1080,
            "format": "JPEG",
            "mode": "RGB",
            "exif": {"camera": "iPhone 14"},
        }
        
        processor = VisionProcessor(config=self.mock_config)
        
        metadata = processor.get_image_metadata(image_data=b"fake_image_data")
        
        self.assertEqual(metadata["width"], 1920)
        self.assertEqual(metadata["height"], 1080)
        self.assertEqual(metadata["format"], "JPEG")


class TestBatchProcessing(unittest.TestCase):
    """Tests for batch image processing."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_config = {
            "vision": {
                "enabled": True,
                "provider": "blip",
            }
        }
    
    @patch("mia.multimodal.vision_processor.VisionProcessor._detect_available_providers")
    @patch("mia.multimodal.vision_processor.VisionProcessor.analyze_image")
    def test_batch_analyze(self, mock_analyze, mock_detect):
        """Test batch image analysis."""
        mock_detect.return_value = {VisionProvider.BLIP}
        mock_analyze.return_value = ImageAnalysisResult(
            caption="Test image",
            confidence=0.9,
            provider="blip",
        )
        
        processor = VisionProcessor(config=self.mock_config)
        
        images = [b"image1", b"image2", b"image3"]
        results = processor.batch_analyze(images)
        
        self.assertEqual(len(results), 3)
        self.assertEqual(mock_analyze.call_count, 3)


if __name__ == "__main__":
    unittest.main()
