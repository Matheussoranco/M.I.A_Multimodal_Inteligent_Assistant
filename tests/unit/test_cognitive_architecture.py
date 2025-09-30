import unittest
from unittest.mock import MagicMock, patch
from mia.core.cognitive_architecture import MIACognitiveCore


class TestMIACognitiveCore(unittest.TestCase):
    """Unit tests for MIACognitiveCore."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_llm = MagicMock()
        self.core = MIACognitiveCore(self.mock_llm, device="cpu")

    def test_init(self):
        """Test initialization."""
        mock_llm = MagicMock()
        core = MIACognitiveCore(mock_llm, device="cpu")
        self.assertIs(core.llm, mock_llm)
        self.assertEqual(core.device, "cpu")
        self.assertIsInstance(core.working_memory, list)

    def test_init_default_device(self):
        """Test initialization with default device."""
        mock_llm = MagicMock()
        with patch('torch.cuda.is_available', return_value=False):
            core = MIACognitiveCore(mock_llm)
            self.assertEqual(core.device, "cpu")

    @patch('torch.cuda.is_available', return_value=True)
    def test_init_cuda_device(self, mock_cuda_available):
        """Test initialization with CUDA device when available."""
        mock_llm = MagicMock()
        core = MIACognitiveCore(mock_llm)
        self.assertEqual(core.device, "cuda")

    @patch('mia.core.cognitive_architecture.HAS_CLIP', True)
    @patch('mia.core.cognitive_architecture.CLIPProcessor')
    @patch('mia.core.cognitive_architecture.CLIPModel')
    def test_init_vision_components_success(self, mock_clip_model, mock_clip_processor):
        """Test successful vision components initialization."""
        mock_processor_instance = MagicMock()
        mock_model_instance = MagicMock()
        mock_clip_processor.from_pretrained.return_value = mock_processor_instance
        mock_clip_model.from_pretrained.return_value = mock_model_instance
        
        # Mock the .to() method to return the same instance
        mock_model_instance.to.return_value = mock_model_instance

        core = MIACognitiveCore(self.mock_llm)
        self.assertEqual(core.vision_processor, mock_processor_instance)
        self.assertEqual(core.vision_model, mock_model_instance)

    @patch('mia.core.cognitive_architecture.HAS_CLIP', False)
    def test_init_vision_components_no_clip(self):
        """Test vision components initialization when CLIP not available."""
        core = MIACognitiveCore(self.mock_llm)
        self.assertIsNone(core.vision_processor)
        self.assertIsNone(core.vision_model)

    @patch('mia.core.cognitive_architecture.SpeechProcessor')
    def test_init_speech_processor_success(self, mock_speech_processor):
        """Test successful speech processor initialization."""
        mock_instance = MagicMock()
        mock_speech_processor.return_value = mock_instance

        core = MIACognitiveCore(self.mock_llm)
        self.assertEqual(core.speech_processor, mock_instance)

    def test_process_multimodal_input_text_only(self):
        """Test processing text-only input."""
        inputs = {
            'text': 'Hello world',
            'image': None,
            'audio': None
        }

        # Mock LLM response
        self.mock_llm.query.return_value = "Processed text response"

        result = self.core.process_multimodal_input(inputs)

        self.assertIn('text', result)
        self.assertIn('embedding', result)
        self.assertEqual(result['text'], "Processed text response")

    def test_process_multimodal_input_with_image(self):
        """Test processing input with image."""
        inputs = {
            'text': 'Describe this image',
            'image': 'path/to/image.jpg',
            'audio': None
        }

        # Mock LLM response
        self.mock_llm.query.return_value = "Image description"

        result = self.core.process_multimodal_input(inputs)

        self.assertIn('text', result)
        self.assertIn('embedding', result)

    def test_process_multimodal_input_with_audio(self):
        """Test processing input with audio."""
        inputs = {
            'text': 'Transcribe this audio',
            'image': None,
            'audio': 'path/to/audio.wav'
        }

        # Mock LLM response
        self.mock_llm.query.return_value = "Audio transcription"

        result = self.core.process_multimodal_input(inputs)

        self.assertIn('text', result)
        self.assertIn('embedding', result)

    def test_process_multimodal_input_multimodal(self):
        """Test processing multimodal input."""
        inputs = {
            'text': 'Analyze this scene',
            'image': 'path/to/image.jpg',
            'audio': 'path/to/audio.wav'
        }

        # Mock LLM response
        self.mock_llm.query.return_value = "Multimodal analysis"

        result = self.core.process_multimodal_input(inputs)

        self.assertIn('text', result)
        self.assertIn('embedding', result)

    def test_process_multimodal_input_empty(self):
        """Test processing empty input."""
        inputs = {}

        result = self.core.process_multimodal_input(inputs)

        self.assertIn('text', result)
        self.assertIn('embedding', result)
        self.assertEqual(result['text'], "No input provided")

    def test_reasoning_pipeline_success(self):
        """Test successful reasoning pipeline."""
        context = "Test context for reasoning"

        # Mock LLM response
        self.mock_llm.query.return_value = "Reasoning result"

        result = self.core._reasoning_pipeline(context)

        self.assertIn('text', result)
        self.assertIn('embedding', result)
        self.assertEqual(result['text'], "Reasoning result")

    def test_reasoning_pipeline_llm_error(self):
        """Test reasoning pipeline with LLM error."""
        context = "Test context"

        # Mock LLM to raise exception
        self.mock_llm.query.side_effect = Exception("LLM error")

        result = self.core._reasoning_pipeline(context)

        self.assertIn('text', result)
        self.assertIn('Error in reasoning pipeline', result['text'])

    def test_reasoning_pipeline_query_model_fallback(self):
        """Test reasoning pipeline with query_model fallback."""
        context = "Test context"

        # Mock llm to not have query method
        del self.mock_llm.query
        self.mock_llm.query_model = MagicMock(return_value="Fallback result")

        result = self.core._reasoning_pipeline(context)

        self.assertEqual(result['text'], "Fallback result")

    def test_reasoning_pipeline_no_llm_methods(self):
        """Test reasoning pipeline when LLM has no query methods."""
        context = "Test context"

        # Mock llm to have no query methods
        self.mock_llm.query = None
        self.mock_llm.query_model = None

        result = self.core._reasoning_pipeline(context)

        self.assertIn('text', result)
        self.assertEqual(result['text'], "")

    def test_generate_embedding_success(self):
        """Test successful embedding generation."""
        text = "Test text for embedding"

        result = self.core._generate_embedding(text)

        # Should return a list (even if empty for mock)
        self.assertIsInstance(result, list)

    def test_generate_embedding_empty_text(self):
        """Test embedding generation with empty text."""
        result = self.core._generate_embedding("")

        self.assertEqual(result, [])

    def test_generate_embedding_whitespace_text(self):
        """Test embedding generation with whitespace-only text."""
        result = self.core._generate_embedding("   ")

        self.assertEqual(result, [])

    def test_generate_embedding_none_text(self):
        """Test embedding generation with None text."""
        result = self.core._generate_embedding(None)

        self.assertEqual(result, [])


if __name__ == "__main__":
    unittest.main()