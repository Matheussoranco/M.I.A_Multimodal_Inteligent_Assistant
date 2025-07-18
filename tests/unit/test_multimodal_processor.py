import unittest
import sys
from pathlib import Path

# Add src directory to path for imports
project_root = Path(__file__).parent.parent.parent
src_dir = project_root / 'src'
sys.path.insert(0, str(src_dir))

try:
    from mia.multimodal.processor import MultimodalProcessor
except ImportError as e:
    print(f"Import error: {e}")
    # Mock the import for testing
    class MultimodalProcessor:
        pass
from unittest.mock import MagicMock, patch

class TestMultimodalProcessor(unittest.TestCase):
    def test_init(self):
        processor = MultimodalProcessor()
        self.assertIsNotNone(processor.recognizer)
        self.assertIsInstance(processor.vision_cache, dict)

    @patch('mia.multimodal.processor.Image.open')
    def test_process_image(self, mock_open):
        processor = MultimodalProcessor()
        mock_img = MagicMock()
        mock_open.return_value = mock_img
        mock_img.size = (100, 100)
        mock_img.__enter__.return_value = mock_img
        mock_img.__exit__.return_value = False
        mock_img.mode = 'RGB'
        processor._get_dominant_color = MagicMock(return_value='red')
        processor._extract_text = MagicMock(return_value='text')
        result = processor.process_image('fake_path.jpg')
        self.assertEqual(result['size'], (100, 100))
        self.assertEqual(result['dominant_color'], 'red')
        self.assertEqual(result['text_ocr'], 'text')

if __name__ == "__main__":
    unittest.main()
