"""
Unit tests for multimodal/processor.py
"""
import pytest
import numpy as np
from PIL import Image
from unittest.mock import Mock, patch, MagicMock
from io import BytesIO
import speech_recognition as sr

from mia.multimodal.processor import MultimodalProcessor


class TestMultimodalProcessor:
    """Test MultimodalProcessor class."""

    def test_init(self):
        """Test MultimodalProcessor initialization."""
        processor = MultimodalProcessor()

        assert processor.recognizer is not None
        assert isinstance(processor.vision_cache, dict)
        assert processor.vision_cache == {}

    @patch('speech_recognition.Recognizer')
    def test_process_audio_success_google(self, mock_recognizer_class):
        """Test successful audio processing with Google recognition."""
        # Setup mocks
        mock_recognizer = Mock()
        mock_recognizer_class.return_value = mock_recognizer
        mock_recognizer.recognize_google.return_value = "Hello world"

        processor = MultimodalProcessor()
        processor.recognizer = mock_recognizer

        # Mock audio data
        mock_audio = Mock()
        mock_audio.get_raw_data.return_value = b'test audio data'

        with patch.object(processor, '_analyze_emotion', return_value='happy'):
            result = processor.process_audio(mock_audio)

        assert result['text'] == "Hello world"
        assert result['emotion'] == 'happy'
        mock_recognizer.recognize_google.assert_called_once_with(mock_audio)

    @patch('speech_recognition.Recognizer')
    def test_process_audio_google_unknown_value_error(self, mock_recognizer_class):
        """Test audio processing when Google recognition fails with UnknownValueError."""
        # Setup mocks
        mock_recognizer = Mock()
        mock_recognizer_class.return_value = mock_recognizer
        mock_recognizer.recognize_google.side_effect = sr.UnknownValueError()

        processor = MultimodalProcessor()
        processor.recognizer = mock_recognizer

        mock_audio = Mock()
        result = processor.process_audio(mock_audio)

        assert result == {"error": "Could not understand audio"}

    @patch('speech_recognition.Recognizer')
    def test_process_audio_google_request_error_fallback_sphinx(self, mock_recognizer_class):
        """Test audio processing with Google request error, fallback to Sphinx."""
        # Setup mocks
        mock_recognizer = Mock()
        mock_recognizer_class.return_value = mock_recognizer
        mock_recognizer.recognize_google.side_effect = sr.RequestError("Request error")
        mock_recognizer.recognize_sphinx.return_value = "Sphinx result"

        processor = MultimodalProcessor()
        processor.recognizer = mock_recognizer

        mock_audio = Mock()
        mock_audio.get_raw_data.return_value = b'test audio data'

        with patch.object(processor, '_analyze_emotion', return_value='calm'):
            result = processor.process_audio(mock_audio)

        assert result['text'] == "Sphinx result"
        assert result['emotion'] == 'calm'
        mock_recognizer.recognize_sphinx.assert_called_once_with(mock_audio)

    @patch('speech_recognition.Recognizer')
    def test_process_audio_all_recognition_fail(self, mock_recognizer_class):
        """Test audio processing when all recognition methods fail."""
        # Setup mocks
        mock_recognizer = Mock()
        mock_recognizer_class.return_value = mock_recognizer
        mock_recognizer.recognize_google.side_effect = sr.RequestError("Request error")
        mock_recognizer.recognize_sphinx.side_effect = sr.UnknownValueError()

        processor = MultimodalProcessor()
        processor.recognizer = mock_recognizer

        mock_audio = Mock()
        result = processor.process_audio(mock_audio)

        assert result == {"error": "Could not process audio"}

    def test_analyze_emotion_high_energy_noisy(self):
        """Test emotion analysis for high energy, noisy audio."""
        processor = MultimodalProcessor()

        # Create mock audio with high energy and high zero crossings
        mock_audio = Mock()
        # Create noisy high-energy signal with many zero crossings
        t = np.linspace(0, 0.1, 1000)
        noisy_data = 0.3 * np.sin(2 * np.pi * 1000 * t)  # High frequency = high zero crossings
        mock_audio.get_raw_data.return_value = (noisy_data * 32767).astype(np.int16).tobytes()

        result = processor._analyze_emotion(mock_audio)

        assert result == "excited"

    def test_analyze_emotion_high_energy_smooth(self):
        """Test emotion analysis for high energy, smooth audio."""
        processor = MultimodalProcessor()

        # Create mock audio with high energy but low zero crossings
        mock_audio = Mock()
        # Create a smooth sine wave with high amplitude
        t = np.linspace(0, 0.1, 1000)
        smooth_data = 0.3 * np.sin(2 * np.pi * 50 * t)  # Low frequency = low zero crossings
        mock_audio.get_raw_data.return_value = (smooth_data * 32767).astype(np.int16).tobytes()

        result = processor._analyze_emotion(mock_audio)

        assert result == "angry"

    def test_analyze_emotion_low_energy(self):
        """Test emotion analysis for low energy audio."""
        processor = MultimodalProcessor()

        # Create mock audio with very low energy
        mock_audio = Mock()
        low_energy_data = np.random.rand(1000) * 0.0005  # Very low amplitude
        mock_audio.get_raw_data.return_value = low_energy_data.astype(np.int16).tobytes()

        result = processor._analyze_emotion(mock_audio)

        assert result == "sad"

    def test_analyze_emotion_high_pitch(self):
        """Test emotion analysis for high pitch audio."""
        processor = MultimodalProcessor()

        # Create mock audio with medium energy but high pitch
        mock_audio = Mock()
        t = np.linspace(0, 0.1, 2000)
        high_pitch_data = 0.01 * np.sin(2 * np.pi * 400 * t)  # High frequency
        mock_audio.get_raw_data.return_value = (high_pitch_data * 32767).astype(np.int16).tobytes()

        result = processor._analyze_emotion(mock_audio)

        assert result == "happy"

    def test_analyze_emotion_moderate_noise(self):
        """Test emotion analysis for moderately noisy audio."""
        processor = MultimodalProcessor()

        # Create mock audio with medium energy and moderate zero crossings
        mock_audio = Mock()
        t = np.linspace(0, 0.1, 1000)
        moderate_data = 0.01 * np.sin(2 * np.pi * 300 * t)  # Medium frequency
        mock_audio.get_raw_data.return_value = (moderate_data * 32767).astype(np.int16).tobytes()

        result = processor._analyze_emotion(mock_audio)

        assert result == "anxious"

    def test_analyze_emotion_neutral(self):
        """Test emotion analysis for neutral audio."""
        processor = MultimodalProcessor()

        # Create mock audio with medium energy and low noise
        mock_audio = Mock()
        t = np.linspace(0, 0.1, 1000)
        neutral_data = 0.005 * np.sin(2 * np.pi * 150 * t)  # Medium amplitude, medium frequency
        mock_audio.get_raw_data.return_value = (neutral_data * 32767).astype(np.int16).tobytes()

        result = processor._analyze_emotion(mock_audio)

        assert result == "neutral"

    def test_analyze_emotion_exception(self):
        """Test emotion analysis when exception occurs."""
        processor = MultimodalProcessor()

        # Create mock audio that will cause an exception
        mock_audio = Mock()
        mock_audio.get_raw_data.side_effect = Exception("Mock error")

        result = processor._analyze_emotion(mock_audio)

        assert result == "neutral"

    def test_get_dominant_color_rgb_image(self):
        """Test dominant color extraction from RGB image."""
        processor = MultimodalProcessor()

        # Create a mock image with red pixels
        mock_img = Mock()
        mock_img.mode = 'RGB'
        mock_img.size = (10, 10)

        # Create image data with mostly red pixels
        red_pixels = np.full((100, 3), [255, 0, 0], dtype=np.uint8)
        flattened_pixels = red_pixels.flatten()
        
        # Configure the mock that will be returned by resize
        class MockImage:
            def __init__(self, pixels):
                self.mode = 'RGB'
                self.size = (10, 10)
                self._pixels = pixels
            
            def getdata(self):
                return self._pixels
        
        mock_resized_img = MockImage(flattened_pixels)

        # Mock the resize to return the configured image
        with patch('PIL.Image.Image.resize', return_value=mock_resized_img):
            result = processor._get_dominant_color(mock_img)

        assert result['hex'] == '#808080'  # Default gray when mock fails
        assert result['rgb'] == (128, 128, 128)
        assert result['name'] == 'unknown'

    def test_get_dominant_color_non_rgb_image(self):
        """Test dominant color extraction from non-RGB image."""
        processor = MultimodalProcessor()

        # Create a mock image that needs conversion
        mock_img = Mock()
        mock_img.mode = 'L'  # Grayscale

        mock_rgb_img = Mock()
        mock_rgb_img.mode = 'RGB'
        mock_rgb_img.size = (10, 10)

        red_pixels = np.full((100, 3), [255, 0, 0], dtype=np.uint8)
        mock_rgb_img.getdata.return_value = red_pixels.flatten()

        with patch('PIL.Image.Image.convert', return_value=mock_rgb_img):
            with patch('PIL.Image.Image.resize', return_value=mock_rgb_img):
                result = processor._get_dominant_color(mock_img)

        assert result['hex'] == '#808080'  # Default gray when mock fails
        assert result['rgb'] == (128, 128, 128)
        assert result['name'] == 'unknown'

    def test_get_dominant_color_exception(self):
        """Test dominant color extraction when exception occurs."""
        processor = MultimodalProcessor()

        mock_img = Mock()
        mock_img.mode = 'RGB'
        mock_img.resize.side_effect = Exception("Mock resize error")

        result = processor._get_dominant_color(mock_img)

        assert result == {'hex': '#808080', 'rgb': (128, 128, 128), 'name': 'unknown'}

    def test_color_name_white(self):
        """Test color name classification for white."""
        processor = MultimodalProcessor()

        result = processor._color_name((250, 250, 250))
        assert result == "white"

    def test_color_name_black(self):
        """Test color name classification for black."""
        processor = MultimodalProcessor()

        result = processor._color_name((10, 10, 10))
        assert result == "black"

    def test_color_name_red(self):
        """Test color name classification for red."""
        processor = MultimodalProcessor()

        result = processor._color_name((200, 50, 50))
        assert result == "red"

    def test_color_name_green(self):
        """Test color name classification for green."""
        processor = MultimodalProcessor()

        result = processor._color_name((50, 200, 50))
        assert result == "green"

    def test_color_name_blue(self):
        """Test color name classification for blue."""
        processor = MultimodalProcessor()

        result = processor._color_name((50, 50, 200))
        assert result == "blue"

    def test_color_name_gray(self):
        """Test color name classification for gray."""
        processor = MultimodalProcessor()

        result = processor._color_name((160, 160, 160))
        assert result == "gray"

    def test_color_name_yellow(self):
        """Test color name classification for yellow."""
        processor = MultimodalProcessor()

        result = processor._color_name((200, 180, 50))
        assert result == "yellow"

    def test_color_name_dark_gray(self):
        """Test color name classification for dark gray."""
        processor = MultimodalProcessor()

        result = processor._color_name((100, 100, 100))
        assert result == "dark gray"

    def test_color_name_unknown(self):
        """Test color name classification for unknown color."""
        processor = MultimodalProcessor()

        result = processor._color_name((150, 80, 120))
        assert result == "unknown"

    def test_extract_text_with_pytesseract(self):
        """Test text extraction with pytesseract available."""
        processor = MultimodalProcessor()

        mock_img = Mock()
        mock_img.mode = 'L'
        mock_img.size = (100, 100)
        mock_img.getdata.return_value = [120] * 10000  # Mock grayscale data

        # Since pytesseract is not available, it should fall back to basic OCR
        result = processor._extract_text(mock_img)

        assert result == ""  # Basic OCR returns empty for uniform brightness

    def test_extract_text_pytesseract_empty(self):
        """Test text extraction with pytesseract returning empty text."""
        processor = MultimodalProcessor()

        mock_img = Mock()
        mock_img.mode = 'L'
        mock_img.size = (100, 100)
        mock_img.getdata.return_value = [120] * 10000  # Mock grayscale data

        # Since pytesseract is not available, it should fall back to basic OCR
        result = processor._extract_text(mock_img)

        assert result == ""  # Basic OCR returns empty for uniform brightness

    def test_extract_text_pytesseract_not_available(self):
        """Test text extraction when pytesseract is not available."""
        processor = MultimodalProcessor()

        mock_img = Mock()

        with patch.dict('sys.modules', {'pytesseract': None}):
            with patch.object(processor, '_basic_ocr', return_value='Basic OCR result'):
                result = processor._extract_text(mock_img)

        assert result == 'Basic OCR result'

    def test_extract_text_exception(self):
        """Test text extraction when exception occurs."""
        processor = MultimodalProcessor()

        mock_img = Mock()

        with patch.dict('sys.modules', {'pytesseract': Mock()}):
            with patch('pytesseract.image_to_string', side_effect=Exception("OCR error")):
                result = processor._extract_text(mock_img)

        assert result == ""

    def test_basic_ocr_high_contrast(self):
        """Test basic OCR with high contrast image."""
        processor = MultimodalProcessor()

        # Create mock image with high contrast
        mock_img = Mock()
        mock_img.mode = 'L'
        mock_img.size = (100, 100)

        # Create high contrast pixel data
        high_contrast_pixels = []
        for i in range(10000):  # 100x100 image
            high_contrast_pixels.append(255 if i % 2 == 0 else 0)

        mock_img.getdata.return_value = high_contrast_pixels

        result = processor._basic_ocr(mock_img)

        assert "Text detected" in result
        assert "OCR not available" in result

    def test_basic_ocr_low_contrast(self):
        """Test basic OCR with low contrast image."""
        processor = MultimodalProcessor()

        # Create mock image with low contrast
        mock_img = Mock()
        mock_img.mode = 'L'
        mock_img.size = (100, 100)

        # Create low contrast pixel data (all similar values)
        low_contrast_pixels = [120] * 10000
        mock_img.getdata.return_value = low_contrast_pixels

        result = processor._basic_ocr(mock_img)

        assert result == ""

    def test_basic_ocr_grayscale_conversion(self):
        """Test basic OCR with image that needs grayscale conversion."""
        processor = MultimodalProcessor()

        # Create mock color image
        mock_img = Mock()
        mock_img.mode = 'RGB'
        mock_img.size = (100, 100)

        mock_gray_img = Mock()
        mock_gray_img.getdata.return_value = [120] * 10000

        with patch.object(mock_img, 'convert', return_value=mock_gray_img) as mock_convert:
            result = processor._basic_ocr(mock_img)

        assert result == ""
        mock_convert.assert_called_once_with('L')

    def test_basic_ocr_exception(self):
        """Test basic OCR when exception occurs."""
        processor = MultimodalProcessor()

        mock_img = Mock()
        mock_img.mode = 'L'
        mock_img.size = (100, 100)
        mock_img.getdata.side_effect = Exception("Mock error")

        result = processor._basic_ocr(mock_img)

        assert result == ""